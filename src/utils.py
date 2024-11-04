import numpy as np
import random
from src.config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    random.seed(seed)

set_seed()