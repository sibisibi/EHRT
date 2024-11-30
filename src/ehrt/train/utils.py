import os
import random
from itertools import chain

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


def split_datasets(dataset, ratios):
    patient_indx = list(dataset.patient_to_index.keys())
    num_patients = len(patient_indx)
    np.random.shuffle(patient_indx)

    data_subsets = []
    idx_start = 0
    idx_end = 0

    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            idx_end = num_patients
        else:
            idx_end += int(num_patients * ratio)

        patients = patient_indx[idx_start: idx_end]
        patient_indices = list(chain(*[dataset.patient_to_index[i] for i in patients]))
        data_subset = torch.utils.data.Subset(dataset, patient_indices)
        data_subsets.append(data_subset)

        idx_start = idx_end

    return data_subsets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())