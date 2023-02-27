from typing import Dict, List

from task_1.validator import FeaturesChoice, HairCutChoices, HairColorChoices, AccessoriesChoices, ClothesTypeChoices,\
    ClothesColorChoice


styles = {FeaturesChoice.haircut: [hair_cut.value for hair_cut in HairCutChoices],
          FeaturesChoice.hair_color: [hair_color.value for hair_color in HairColorChoices],
          FeaturesChoice.accessories: [accessories.value for accessories in AccessoriesChoices],
          FeaturesChoice.clothes_type: [clothes_type.value for clothes_type in ClothesTypeChoices],
          FeaturesChoice.clothes_color: [clothes_color.value for clothes_color in ClothesColorChoice]
          }

styles_count = {FeaturesChoice.haircut: [7, 0, 1, 23, 1, 11, 7],
                FeaturesChoice.hair_color: [7, 6, 2, 3, 8, 24],
                FeaturesChoice.accessories: [11, 22, 17],
                FeaturesChoice.clothes_type: [7, 18, 19, 6],
                FeaturesChoice.clothes_color: [4, 5, 6, 8, 6, 8, 7, 6]
                }


def create_dataset() -> Dict[FeaturesChoice, List[float]]:

    collected_data = {}

    for style_name, style_list in styles.items():
        collected_data[style_name] = []
        total_elements = len(styles_count[style_name])
        for feature_name, feature_count in zip(styles[style_name], styles_count[style_name]):
            collected_data[style_name].append((feature_count + 1) / (sum(styles_count[style_name]) + total_elements))

    return collected_data
