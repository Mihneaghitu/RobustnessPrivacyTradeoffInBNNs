from enum import Enum


class AttackType(Enum):
    FGSM = 1
    PGD = 2
    IBP = 3