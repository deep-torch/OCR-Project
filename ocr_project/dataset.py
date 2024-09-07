import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.utils.rnn as rnn_utils
from torchvision import transforms

import numpy as np
from PIL import Image


class IAMDataset(Dataset):
    def __init__(self, root_dir, annotation_file, token_file='tokens.json', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the annotations file with labels.
            token_file (string): Path to the JSON file with token to index mapping.
            transform: Optional transform.
        """
        self.root_dir = root_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform

        with open(token_file) as f:
            self.token_to_index = json.load(f)
            self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def _load_annotations(self, annotation_file):
        annotations = []
        with open(annotation_file) as file:
            for line in file:
                parts = line.strip().split(' ')
                image_path = parts[0]
                label = parts[-1]
                annotations.append((image_path, label))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def get_image_path(self, idx):
        full = self.annotations[idx][0]
        full2 = full.strip().split('-')
        p1 = full2[0]
        p2 = p1 + '-' + full2[1]
        p3 = full
        return os.path.join(self.root_dir, p1, p2, p3 + '.jpg')

    def __getitem__(self, idx):
        img_name = self.get_image_path(idx)

        try:
            image = Image.open(img_name).convert('L')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self.annotations))

        label = self.annotations[idx][1]

        if self.transform:
            image = self.transform(image)

        label_encoded = [self.token_to_index[char] for char in label]
        return image, torch.tensor(label_encoded, dtype=torch.long), len(label_encoded)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels_padded = rnn_utils.pad_sequence(labels, batch_first=True, padding_value=-1)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels_padded, lengths


def split_data(root_dir, annotation_file, token_file, train_ratio=0.8, seed=42):
    full_dataset = IAMDataset(root_dir=root_dir, annotation_file=annotation_file, token_file=token_file)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(indices)

    train_size = int(train_ratio * dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


def get_dataloaders(root_dir, annotation_file, token_file, batch_size, num_workers=2, seed=42):
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset, test_dataset = split_data(root_dir, annotation_file, token_file, seed=seed)

    train_dataset.dataset.transform = train_transform
    test_dataset.dataset.transform = test_transform

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_dataloader, test_dataloader
