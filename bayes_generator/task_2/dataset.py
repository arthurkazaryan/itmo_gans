from pathlib import Path
from typing import Dict
import random

import numpy as np
import cv2


class ImageBayesDataset:
    dataset: Dict[str, np.ndarray] = {}
    dataset_probability: Dict[str, np.ndarray] = {}
    height: int = None
    width: int = None

    _is_calculated_probability = False

    def fit(self, image_path: Path) -> None:
        array = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if not self.height:
            self.height, self.width = array.shape
        for row_id in range(array.shape[-1]):
            for hei_id in range(array[:, row_id].shape[-1]):
                current_pix = f"{hei_id}_{row_id}"
                if current_pix not in self.dataset:
                    self.dataset[current_pix] = np.zeros((256,), dtype=np.uint8)
                self.dataset[current_pix][array[hei_id, row_id]] += 1

    def calculate_probability(self) -> None:

        for pixel_id, pixel_data in self.dataset.items():
            current_pixel_probability = []
            sum_of_pixel_data = sum(pixel_data)
            for pixel_count in pixel_data:
                current_pixel_probability.append(pixel_count / sum_of_pixel_data)
            self.dataset_probability[pixel_id] = np.array(current_pixel_probability)

        self._is_calculated_probability = True

    def generate(self) -> np.ndarray:
        if not self._is_calculated_probability:
            self.calculate_probability()

        output_image = np.zeros((self.height, self.width))
        pixel_range = np.arange(256)

        for hei_id in range(self.height):
            for row_id in range(self.width):
                output_image[hei_id, row_id] = random.choices(pixel_range,
                                                              self.dataset_probability[f'{hei_id}_{row_id}'])[0]

        return output_image
