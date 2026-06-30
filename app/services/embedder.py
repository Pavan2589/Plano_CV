"""
ResNet50 product embedder.

Generates a 2048-d, L2-normalized feature vector for a cropped product
image. Classification head is removed (replaced with Identity) so the
raw feature vector is returned instead of a 1000-class softmax.
"""
import time
import logging

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

logger = logging.getLogger("plano_cv.embedder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)


class ProductEmbedder:
    def __init__(self):
        logger.info("Initializing ResNet50 embedder...")
        start = time.time()

        try:
            weights = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        except AttributeError:
            self.model = models.resnet50(pretrained=True)

        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"ResNet50 embedder ready in {time.time() - start:.3f}s")

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(tensor).squeeze(0)
            norm = torch.norm(features, p=2, keepdim=True)
            normalized = features / (norm + 1e-12)
            embedding = normalized.cpu().numpy()

        if embedding.shape[0] != 2048:
            raise ValueError(f"Expected 2048-d embedding, got {embedding.shape[0]}")

        return embedding


embedder_instance = ProductEmbedder()
