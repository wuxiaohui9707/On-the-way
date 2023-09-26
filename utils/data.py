#from custom_datasets import CustomDataset
from . import custom_datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_loaders(raw_datasets,transform, batch_size):
    train_dataset = custom_datasets.CustomDataset(raw_datasets['train'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = custom_datasets.CustomDataset(raw_datasets['validation'], transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = custom_datasets.CustomDataset(raw_datasets['test'], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   

    return train_loader, val_loader, test_loader
