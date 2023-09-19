from torch.utils.data import Dataset
import torch

###
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        label = float(item["label"])  # 标签是连续值

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
