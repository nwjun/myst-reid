import cv2
import numpy as np
from PIL import Image


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def __call__(self, img):
        """
        img: PIL.Image
        return: PIL.Image
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Convert PIL → numpy
        img_np = np.array(img)

        # Handle grayscale
        if img_np.ndim == 2:
            img_np = self.clahe.apply(img_np)
            return Image.fromarray(img_np)

        # Handle RGB
        if img_np.ndim == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)

            l = self.clahe.apply(l)

            lab = cv2.merge((l, a, b))
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_np)

        raise ValueError("Unsupported image format")
