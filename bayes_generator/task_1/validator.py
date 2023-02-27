from enum import Enum

from pydantic import BaseModel


class HairCutChoices(Enum):
    no_hair = "нет волос"
    long_ponytail = "длинные в пучок"
    long_wave = "длинные волнистые"
    long_straight = "длинные прямые"
    short_wave = "короткая волнистые"
    short_straight = "короткая прямые"
    short_curly = "короткая курчавые"


class HairColorChoices(Enum):
    black = "черный"
    blonde = "блонд"
    chestnut = "каштановый"
    pastel_pink = "пастельный розовый"
    ginger = "рыжий"
    silver = "серебристо серый"


class AccessoriesChoices(Enum):
    no_glasses = "нет очков"
    round_glasses = "круглые очки"
    sun_glasses = "солнцезащитные очки"


class ClothesTypeChoices(Enum):
    hoodie = "худи"
    overall = "комбинезон"
    t_shirt_round_neck = "футболка с круглым вырезом"
    t_shirt_v_neck = "футболка с V-вырезом"


class ClothesColorChoice(Enum):
    black = "черный"
    blue = "синий"
    gray = "серый"
    green = "зеленый"
    orange = "оранжевый"
    pink = "розовый"
    red = "красный"
    white = "белый"


class FeaturesChoice(Enum):
    haircut = 'прическа'
    hair_color = 'цвет волос'
    accessories = 'аксессуар'
    clothes_type = 'одежда'
    clothes_color = 'цвет одежды'


class ClothesValidator(BaseModel):
    haircut: HairCutChoices
    hair_color: HairColorChoices
    accessories: AccessoriesChoices
    clothes_type: ClothesTypeChoices
    clothes_color: ClothesColorChoice

    class Config:
        use_enum_values = True
