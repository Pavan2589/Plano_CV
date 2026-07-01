"""
Draws compliance check results onto the shelf image.
Green = matched, Red = wrong product, Purple = unexpected, legend for missing.
"""
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("plano_cv.annotator")
logger.setLevel(logging.INFO)


class ResultAnnotator:

    def annotate(self, image: Image.Image, match_result: dict) -> Image.Image:
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        drawn_label_positions = []

        def draw_label(bbox, text, color):
            draw.rectangle(bbox, outline=color, width=3)
            label_x, label_y = bbox[0], max(0, bbox[1] - 14)
            for prev_x, prev_y in drawn_label_positions:
                if abs(label_x - prev_x) < 70 and abs(label_y - prev_y) < 14:
                    label_y = max(0, label_y - 14)
            draw.text((label_x, label_y), text, fill=color, font=font)
            drawn_label_positions.append((label_x, label_y))

        for m in match_result.get("matched", []):
            draw_label(m["detected_bbox"], f"{m['product_name']} {m['similarity']:.2f}", "green")

        for w in match_result.get("wrong_product", []):
            draw_label(w["detected_bbox"], f"Expected {w['expected_product_name']} ({w['similarity']:.2f})", "red")

        for u in match_result.get("unexpected", []):
            draw_label(u["bbox"], "Unexpected", "purple")

        missing = match_result.get("missing", [])
        if missing:
            legend_x, legend_y = 10, 10
            line_height = 16
            box_height = 30 + len(missing) * line_height
            draw.rectangle([legend_x, legend_y, legend_x + 320, legend_y + box_height],
                            fill=(0, 0, 0, 180), outline="orange", width=2)
            draw.text((legend_x + 10, legend_y + 5), "Missing Products:", fill="white", font=font)
            for idx, miss in enumerate(missing):
                draw.text((legend_x + 10, legend_y + 25 + idx * line_height),
                          f"- {miss['product_name']}", fill="orange", font=font)

        for tag in match_result.get("price_tags_detected", []):
            bbox = tag["bbox"]
            draw.rectangle(bbox, outline="cyan", width=2)

        for group in match_result.get("missing_price_tags", []):
            bbox = group["group_bbox"]
            draw.rectangle(bbox, outline="orange", width=2)
            label = f"{group['product_name']} - no price tag ({group['item_count']} units)"
            label_x, label_y = bbox[0], max(0, bbox[3] + 4)
            draw.text((label_x, label_y), label, fill="orange", font=font)

        return annotated

    def save(self, image: Image.Image, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="JPEG")
        return output_path


annotator_instance = ResultAnnotator()
