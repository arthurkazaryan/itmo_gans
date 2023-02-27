import random

from task_1.dataset import create_dataset, styles
from task_1.validator import ClothesValidator


collected_dataset = create_dataset()


def generate_data() -> ClothesValidator:

    gen_data = {}
    for style_name, style_data in collected_dataset.items():
        gen_data.update({style_name.name: random.choices(styles[style_name], style_data)[0]})

    validated_data = ClothesValidator(**gen_data)

    return validated_data


generated_data = generate_data()
print(generated_data.dict())
