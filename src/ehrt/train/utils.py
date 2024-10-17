import os
import random

import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Seed the PyTorch CUDA generator for single-GPU
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs (if using multi-GPU setup)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())