import csv
import os
import random


import torch
import torchvision
from typing import Optional, Callable, Any


image_size = 64


def _to_float_values(row):
    labels = []

    if row[1] == "Male":
        labels.append(0.0)
    else:
        labels.append(1.0)
    
    labels.append(float(row[2]))

    if row[3] == "Frontal":
        labels.append(0.0)
    else:
        labels.append(1.0)

    if row[4] == "AP":
        labels.append(0.0)
    else:
        labels.append(1.0)

    if row[5] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[5]))

    if row[6] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[6]))

    if row[7] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[7]))

    if row[8] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[8]))

    if row[9] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[9]))

    if row[10] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[10]))

    if row[11] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[11]))

    if row[12] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[12]))

    if row[13] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[13]))

    if row[14] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[14]))

    if row[15] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[15]))

    if row[16] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[16]))

    if row[17] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[17]))

    if row[18] == "":
        labels.append(0.0)
    else:
        labels.append(float(row[18]))

    return labels


class ChexpertSmallV1(torchvision.datasets.VisionDataset):
    """
        The small version of the CheXpert dataset, ready to be used for supervised learning
    """
    def __init__(self, root: str, max_images: Optional[int] = None, validation: bool = False, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.resizeTransform = torchvision.transforms.Resize((image_size, image_size))
        self.toPIL = torchvision.transforms.ToPILImage()

        if validation:
            label_file = "valid.csv"
        else:
            label_file = "train.csv"

        with open(os.path.join(root, "CheXpert-v1.0-small", label_file)) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            self.labelNames = next(reader)
            self.num_classes = len(self.labelNames) - 1
            self.labels = []
            self.imageLabels = []
            for row in reader:
                self.imageLabels.append(row[0])
                self.labels.append(_to_float_values(row))

        if max_images is not None:
            self.max_images = max_images
        else:
            self.max_images = len(self.imageLabels)

    def __len__(self) -> int:
        return min(self.max_images, len(self.imageLabels))

    def __getitem__(self, index: int) -> Any:
        img_path = os.path.join(self.root, self.imageLabels[index])
        image = torchvision.io.read_image(img_path)
        image = image.repeat(3, 1, 1)
        image = self.toPIL(image)
        image = self.resizeTransform(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            image = self.target_transform(image)
        return image, torch.tensor(self.labels[index])


class ChexpertSmallV1MICLE(torchvision.datasets.VisionDataset):
    """
        The small version of the CheXpert dataset, ready to be used for multi instance contrastive learning (MICLe)
    """
    def __init__(self, root: str, max_images: Optional[int] = None) -> None:
        super().__init__(root, None, None, None)

        self.imagePaths = {}
        self.labels = {}
        
        with open(os.path.join(root, "CheXpert-v1.0-small", "train.csv")) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            self.labelNames = next(reader)
            self.num_classes = len(self.labelNames) - 1
            for row in reader:
                patient_index = self._extract_patient_index(row)
                if patient_index not in self.imagePaths:
                    self.imagePaths[patient_index] = []
                    self.labels[patient_index] = _to_float_values(row)
                self.imagePaths[patient_index].append(row[0])

        color_jitter = torchvision.transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2
        )
        self.random_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((image_size, image_size)),
                torchvision.transforms.RandomResizedCrop(size=(image_size, image_size)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomApply([color_jitter], p=0.8),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.max_images = max_images

    def _extract_patient_index(self, row):
        return int(row[0].split("/")[2][7:]) - 1

    def __len__(self) -> int:
        return min(self.max_images, len(self.imagePaths))

    def __getitem__(self, index: int) -> Any:
        imagePaths = self.imagePaths[index]
        label = self.labels[index]

        if len(imagePaths) > 1:
            [index1, index2] = random.sample(range(len(imagePaths)), k=2)
            img1, img2 = imagePaths[index1], imagePaths[index2]
        else:
            img1, img2 = imagePaths[0], imagePaths[0]

        img1, img2 = torchvision.io.read_image(img1), torchvision.io.read_image(img2)
        img1, img2 = img1.repeat(3, 1, 1), img2.repeat(3, 1, 1)
        img1, img2 = self.random_transform(img1), self.random_transform(img2)
        return (img1, img2), torch.tensor(label)
