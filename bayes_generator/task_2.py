from pathlib import Path

import cv2

from task_2.dataset import ImageBayesDataset


image_bayes_dataset = ImageBayesDataset()

for image_path in Path('./task_2/images').glob('*.png'):
    print(image_path)
    image_bayes_dataset.fit(image_path)

image_bayes_dataset.calculate_probability()

for i in range(3):
    generated_image = image_bayes_dataset.generate()
    cv2.imwrite(f'./task_2/test_image_{i}.png', generated_image)
