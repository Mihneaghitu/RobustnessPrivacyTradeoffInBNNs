from enum import Enum

import torch


class LoggerType(Enum):
    STDOUT = 1
    WANDB = 2

cuda_available = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda' if cuda_available else 'cpu')
LOGGER_TYPE = LoggerType.STDOUT
ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING = 42
torch.manual_seed(ANSWER_TO_THE_ULTIMATE_QUESTION_OF_LIFE_THE_UNIVERSE_AND_EVERYTHING)
