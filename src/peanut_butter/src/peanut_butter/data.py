import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class PhraseDataset(Dataset):
    """PyTorch Dataset wrapper for the extracted JSON phrases."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_raw_wmt14(num_samples: int):
    """Streams the WMT14 dataset for the extraction phase."""
    print(f"--- [Data] Loading WMT14 (subset: {num_samples}) ---")
    return load_dataset("wmt14", "de-en", split=f"train[:{num_samples}]", trust_remote_code=True)
