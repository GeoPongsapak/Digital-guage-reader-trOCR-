from os import getcwd
import sys
sys.path.append(getcwd())
from dataclasses import dataclass
from os.path import join
from typing import Any

@dataclass(frozen=True)
class GENERAL_CONFIG:
    SOURCE_FOLDER: str = "src/"
    UTILITY_FOLDERL: str = "utils/"
    IMAGES_FOLDER: str = "images/"
    MODEL_FOLDER: str = "model/"
    CONFIDENCE: float = 0.1
    PROCESSOR_MODEL : str = 'microsoft/trocr-small-printed'