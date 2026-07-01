from typing import List, Dict, Any
import logging

logger = logging.getLogger("plano_cv.position_assigner")
logger.setLevel(logging.INFO)


class PositionAssigner:

    def assign(self, detections: List[Dict[str, Any]],
               rail_y_positions: List[float],
               image_height: float) -> List[Dict[str, Any]]:
        """
        Assigns (row, position) to each detection using:
        - row: which band between shelf rails the product's center_y falls in
        - position: left-to-right index within that row, sorted by x1

        Rail Y positions divide the image into bands:
          Band 1 (Row 1): y=0 to first rail
          Band 2 (Row 2): first rail to second rail
          Band 3 (Row 3): second rail to third rail
          etc.
          Last band: last rail to image bottom

        Returns the same detections list with row and position fields added.
        Does not modify the input list — returns a new list.
        """
        # Build row band boundaries from rail positions
        boundaries = [0.0] + list(rail_y_positions) + [image_height]
        bands = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        # Assign each detection to a row band using center_y
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            center_y = (y1 + y2) / 2.0
            center_x = (x1 + x2) / 2.0
            det["_center_y"] = center_y
            det["_center_x"] = center_x
            det["row"] = None

            for band_idx, (top, bottom) in enumerate(bands):
                if top <= center_y < bottom:
                    det["row"] = band_idx + 1
                    break

            # Fallback: assign to last row if center_y equals image_height exactly
            if det["row"] is None:
                det["row"] = len(bands)

        # Within each row, sort by x1 and assign position left-to-right
        from collections import defaultdict
        rows = defaultdict(list)
        for det in detections:
            rows[det["row"]].append(det)

        result = []
        for row_num in sorted(rows.keys()):
            row_dets = sorted(rows[row_num], key=lambda d: d["bbox"][0])
            for pos_idx, det in enumerate(row_dets):
                det_copy = dict(det)
                det_copy["position"] = pos_idx + 1
                # Clean up internal fields
                det_copy.pop("_center_y", None)
                det_copy.pop("_center_x", None)
                result.append(det_copy)

        logger.info(f"Position assignment complete: "
                    f"{len(result)} detections across {len(rows)} rows")
        for row_num in sorted(rows.keys()):
            row_items = [d for d in result if d["row"] == row_num]
            logger.info(f"  Row {row_num}: {len(row_items)} products — "
                        f"{[d.get('matched_product', d.get('product_name', '?')) for d in row_items]}")

        return result


position_assigner_instance = PositionAssigner()
