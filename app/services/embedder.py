"""
CLIP product embedder (drop-in replacement for the ResNet50 embedder).

Generates an L2-normalized feature vector for a cropped product image
using CLIP's vision tower. ViT-B-32 gives a 512-d embedding and is fast
enough for CPU inference; switch to ViT-L-14 (768-d) if you need more
discriminative power and can afford the extra latency/GPU.

Requires: pip install open_clip_torch
"""
import time
import logging

import torch
import open_clip
from PIL import Image
import numpy as np

logger = logging.getLogger("plano_cv.embedder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

# Swap to ("ViT-L-14", "openai") or ("ViT-L-14", "laion2b_s32b_b82k") for a
# stronger but slower/heavier model.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EXPECTED_DIM = 512


class ProductEmbedder:
    def __init__(self):
        logger.info(f"Initializing CLIP embedder ({MODEL_NAME}/{PRETRAINED})...")
        start = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"CLIP embedder ready in {time.time() - start:.3f}s on {self.device}")

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor).squeeze(0)
            norm = torch.norm(features, p=2, keepdim=True)
            normalized = features / (norm + 1e-12)
            embedding = normalized.cpu().numpy()

        if embedding.shape[0] != EXPECTED_DIM:
            raise ValueError(f"Expected {EXPECTED_DIM}-d embedding, got {embedding.shape[0]}")

        return embedding


embedder_instance = ProductEmbedder()