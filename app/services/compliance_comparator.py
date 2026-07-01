import os
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

logger = logging.getLogger("plano_cv.compliance_comparator")
logger.setLevel(logging.INFO)


class ComplianceComparator:
    def __init__(self):
        self.count_weight = float(os.getenv("COUNT_WEIGHT", "0.5"))
        self.sequence_weight = float(os.getenv("SEQUENCE_WEIGHT", "0.5"))

    def _get_row_sequences(self, items: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        Groups items by row number and returns a dict of
        row_number -> [product_name, product_name, ...]
        sorted by position within each row.
        """
        from collections import defaultdict
        rows = defaultdict(list)
        for item in items:
            row = item.get("row")
            position = item.get("position")
            product = item.get("product_name") or item.get("matched_product")
            if row is not None and position is not None and product:
                rows[row].append((position, product))

        return {
            row_num: [p for _, p in sorted(pairs, key=lambda x: x[0])]
            for row_num, pairs in rows.items()
        }

    def _pad_shelf_sequence(self, ref_seq: List[str],
                             shelf_seq: List[str]) -> List[str]:
        """
        When product counts differ between reference and shelf sequences,
        pad the shelf sequence with "-----" (empty slot) markers at the
        correct positions for each missing product.

        We already know WHICH product is short from the count comparison,
        so we find where that product's group ends in the shelf sequence
        and insert the empty slot right after the last occurrence of
        that product in the shelf sequence.

        Handles multiple missing products across different product types.
        Returns the padded shelf sequence, same length as ref_seq.
        """
        ref_counts = Counter(ref_seq)
        shelf_counts = Counter(shelf_seq)

        padded = list(shelf_seq)

        for product, ref_count in ref_counts.items():
            shelf_count = shelf_counts.get(product, 0)
            shortage = ref_count - shelf_count

            if shortage <= 0:
                continue

            for _ in range(shortage):
                # Find the last occurrence of this product in padded sequence
                last_idx = None
                for i, p in enumerate(padded):
                    if p == product:
                        last_idx = i

                if last_idx is not None:
                    # Insert empty slot right after the last occurrence
                    padded.insert(last_idx + 1, "-----")
                else:
                    # Product not found at all in shelf sequence
                    # Find where it should be based on reference position
                    ref_first_idx = ref_seq.index(product)
                    insert_idx = min(ref_first_idx, len(padded))
                    padded.insert(insert_idx, "-----")

        return padded

    def _compare_sequences(self, ref_seq: List[str],
                            shelf_seq: List[str],
                            row_num: int) -> Dict[str, Any]:
        """
        Compares two sequences of equal length position by position.
        Returns per-position violations and a sequence accuracy score.
        """
        violations = []
        matches = 0
        total = len(ref_seq)

        for pos_idx, (ref_product, shelf_product) in enumerate(
                zip(ref_seq, shelf_seq)):
            position = pos_idx + 1

            if shelf_product == "-----":
                violations.append({
                    "type": "missing_product",
                    "row": row_num,
                    "position": position,
                    "expected_product": ref_product,
                    "actual_product": None,
                })
            elif ref_product == shelf_product:
                matches += 1
            else:
                violations.append({
                    "type": "wrong_product",
                    "row": row_num,
                    "position": position,
                    "expected_product": ref_product,
                    "actual_product": shelf_product,
                })

        sequence_score = matches / total if total > 0 else 1.0
        return {
            "violations": violations,
            "matches": matches,
            "total": total,
            "sequence_score": round(sequence_score, 4),
        }

    def compare(self, reference_items: List[Dict[str, Any]],
                shelf_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main comparison method.

        1. Groups both reference and shelf items into rows
        2. For each row, compares product counts
        3. If counts match: direct sequence comparison
        4. If counts differ: pad shelf sequence with empty slots
           for missing products, then compare
        5. Computes count score, sequence score, and overall score
        """
        ref_rows = self._get_row_sequences(reference_items)
        shelf_rows = self._get_row_sequences(shelf_items)

        all_row_numbers = sorted(set(list(ref_rows.keys()) +
                                     list(shelf_rows.keys())))

        count_violations = []
        sequence_violations = []
        unexpected_products = []
        row_results = []

        total_count_score = 0.0
        total_sequence_score = 0.0
        scored_rows = 0

        for row_num in all_row_numbers:
            ref_seq = ref_rows.get(row_num, [])
            shelf_seq = shelf_rows.get(row_num, [])

            if not ref_seq and shelf_seq:
                # Entire row exists on shelf but not in planogram
                for pos_idx, product in enumerate(shelf_seq):
                    unexpected_products.append({
                        "type": "unexpected_product",
                        "row": row_num,
                        "position": pos_idx + 1,
                        "actual_product": product,
                    })
                continue

            if ref_seq and not shelf_seq:
                # Entire row missing from shelf
                for pos_idx, product in enumerate(ref_seq):
                    sequence_violations.append({
                        "type": "missing_product",
                        "row": row_num,
                        "position": pos_idx + 1,
                        "expected_product": product,
                        "actual_product": None,
                    })
                row_results.append({
                    "row": row_num,
                    "count_score": 0.0,
                    "sequence_score": 0.0,
                    "note": "entire row missing from shelf"
                })
                total_count_score += 0.0
                total_sequence_score += 0.0
                scored_rows += 1
                continue

            # Count comparison per product per row
            ref_counts = Counter(ref_seq)
            shelf_counts = Counter(shelf_seq)
            all_products = set(list(ref_counts.keys()) +
                               list(shelf_counts.keys()))

            row_count_scores = []
            counts_match = True

            for product in all_products:
                expected = ref_counts.get(product, 0)
                actual = shelf_counts.get(product, 0)

                if expected > 0:
                    product_count_score = min(actual, expected) / expected
                    row_count_scores.append(product_count_score)

                if actual > expected and expected > 0:
                    unexpected_products.append({
                        "type": "unexpected_product",
                        "row": row_num,
                        "product": product,
                        "expected_count": expected,
                        "actual_count": actual,
                        "extra_count": actual - expected,
                    })

                if actual != expected:
                    counts_match = False
                    if actual < expected:
                        count_violations.append({
                            "type": "count_mismatch",
                            "row": row_num,
                            "product": product,
                            "expected_count": expected,
                            "actual_count": actual,
                            "short_by": expected - actual,
                        })

            row_count_score = (sum(row_count_scores) / len(row_count_scores)
                               if row_count_scores else 0.0)

            # Sequence comparison
            if counts_match:
                seq_result = self._compare_sequences(
                    ref_seq, shelf_seq, row_num
                )
            else:
                padded_shelf = self._pad_shelf_sequence(ref_seq, shelf_seq)
                seq_result = self._compare_sequences(
                    ref_seq, padded_shelf, row_num
                )

            sequence_violations.extend(seq_result["violations"])

            total_count_score += row_count_score
            total_sequence_score += seq_result["sequence_score"]
            scored_rows += 1

            row_results.append({
                "row": row_num,
                "reference_sequence": ref_seq,
                "shelf_sequence": shelf_seq,
                "counts_matched": counts_match,
                "count_score": round(row_count_score, 4),
                "sequence_score": seq_result["sequence_score"],
                "matches": seq_result["matches"],
                "total_positions": seq_result["total"],
            })

        avg_count_score = (total_count_score / scored_rows
                           if scored_rows > 0 else 0.0)
        avg_sequence_score = (total_sequence_score / scored_rows
                              if scored_rows > 0 else 0.0)
        overall_score = round(
            (self.count_weight * avg_count_score +
             self.sequence_weight * avg_sequence_score) * 100, 2
        )

        logger.info(f"Compliance comparison complete: "
                    f"overall_score={overall_score}%, "
                    f"count_score={avg_count_score:.3f}, "
                    f"sequence_score={avg_sequence_score:.3f}")

        return {
            "overall_score": overall_score,
            "count_score": round(avg_count_score * 100, 2),
            "sequence_score": round(avg_sequence_score * 100, 2),
            "count_violations": count_violations,
            "sequence_violations": sequence_violations,
            "unexpected_products": unexpected_products,
            "row_results": row_results,
        }


compliance_comparator_instance = ComplianceComparator()
