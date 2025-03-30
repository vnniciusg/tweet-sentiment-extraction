from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader


class SentimentDataset(Dataset):
    def __init__(self, encodings, device, labels=None):
        self.encodings = encodings
        self.device = device
        self.labels = labels

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: val[idx].to(self.device).clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        return item


def create_dataloader(dataset: Dataset, batch_size: Optional[int] = 16, shuffle: Optional[bool] = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(42),
    )
