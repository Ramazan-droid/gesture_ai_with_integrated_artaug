import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class OpenCVImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.class_to_idx = {
            cls_name: i for i, cls_name in enumerate(self.classes)
        }

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_dir):
                for file in sorted(files):
                    if file.lower().endswith(valid_extensions):
                        self.image_paths.append(
                            os.path.join(root, file)
                        )
                        self.labels.append(
                            self.class_to_idx[class_name]
                        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            raise FileNotFoundError(
                f"Failed to read image at: {img_path}"
            )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            image_tensor = self.transform(img_rgb)
        else:
            image_tensor = torch.from_numpy(
                img_rgb.transpose((2, 0, 1))
            ).float() / 255.0

        label = self.labels[idx]

        return image_tensor, label



MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def opencv_train_transform(image):

    
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    
    angle = random.uniform(-15, 15)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    image = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    
    image = image.astype(np.float32) / 255.0

    
    image = (image - MEAN) / STD

    
    image = image.transpose((2, 0, 1))

    return torch.from_numpy(image)


def opencv_test_transform(image):

    
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    
    image = image.astype(np.float32) / 255.0

    
    image = (image - MEAN) / STD

    
    image = image.transpose((2, 0, 1))

    return torch.from_numpy(image)


class TransferLearningModel(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        
        for param in self.model.parameters():
            param.requires_grad = False

        
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)



raw_data_path = r"C:\Users\ramaz\Desktop\compvision\asik1\data\raw"

full_dataset = OpenCVImageDataset(raw_data_path)

print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)


train_size = int(0.70 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

train_indices, val_indices, test_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size, test_size],
    generator=generator
)


train_full = OpenCVImageDataset(
    raw_data_path,
    transform=opencv_train_transform
)

val_full = OpenCVImageDataset(
    raw_data_path,
    transform=opencv_test_transform
)

test_full = OpenCVImageDataset(
    raw_data_path,
    transform=opencv_test_transform
)


train_dataset = torch.utils.data.Subset(
    train_full,
    train_indices.indices
)

val_dataset = torch.utils.data.Subset(
    val_full,
    val_indices.indices
)

test_dataset = torch.utils.data.Subset(
    test_full,
    test_indices.indices
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

valid_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
 

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Training on device: {device}")


model = TransferLearningModel(
    num_classes=3
).to(device)

print(model)


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.model.fc.parameters(),
    lr=0.001
)


num_epochs = 10


for epoch in range(num_epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = 100 * correct / total
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {running_loss / len(train_loader):.4f}, "
        f"Accuracy: {accuracy:.2f}%",
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )
torch.save(model.state_dict(), "trslearningmodel.pth")
print("Model weights saved to")



model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()

test_accuracy = 100 * test_correct / test_total

print()
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
