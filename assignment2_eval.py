
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 1. CONFIGURATION
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

RAW_DATA_PATH = r"C:\Users\ramaz\Desktop\asik2\data\raw"

A1_WEIGHTS = r"C:\Users\ramaz\Desktop\asik2\trslearningmodel.pth"
A2_WEIGHTS = r"C:\Users\ramaz\Desktop\asik2\SOTA_model_weights.pth"

BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)



# 2. DATASET

class OpenCVImageDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
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
            raise FileNotFoundError(img_path)

        img_rgb = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            img_rgb,
            (224, 224),
            interpolation=cv2.INTER_LINEAR
        )

        image = image.astype(np.float32) / 255.0

        image = (image - MEAN) / STD

        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)

        label = self.labels[idx]

        return image, label


# 3. MODEL ARCHITECTURE

class TransferLearningModel(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# 4. RECREATE ORIGINAL TEST SPLIT

full_dataset = OpenCVImageDataset(RAW_DATA_PATH)

print("Classes:", full_dataset.classes)
print("Total images:", len(full_dataset))

train_size = int(0.70 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

generator = torch.Generator().manual_seed(42)

_, _, test_indices = torch.utils.data.random_split(
    range(len(full_dataset)),
    [train_size, val_size, test_size],
    generator=generator
)

test_dataset = Subset(
    full_dataset,
    test_indices.indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test images:", len(test_dataset))


# 5. MODEL EVALUATION FUNCTION

def evaluate_checkpoint(weights_path):

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    model = TransferLearningModel(
        num_classes=len(full_dataset.classes)
    ).to(device)

    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }, all_labels, all_predictions


# 6. EVALUATE A2 MODEL

print("\n========== A2 MODEL EVALUATION ==========")

a2_metrics, y_true, y_pred = evaluate_checkpoint(
    A2_WEIGHTS
)

for metric, value in a2_metrics.items():

    print(f"{metric}: {value * 100:.2f}%")


# 7. CLASSIFICATION REPORT

report = classification_report(
    y_true,
    y_pred,
    target_names=full_dataset.classes,
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)


# 8. CONFUSION MATRIX

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=full_dataset.classes
)

display.plot(
    cmap="Blues",
    ax=ax,
    values_format="d"
)

plt.title("ResNet18 - Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# 9. A1 VS A2 ABLATION COMPARISON

print("\n========== A1 VS A2 ABLATION ==========")

experiments = {
    "A1 - Original Dataset": A1_WEIGHTS,
    "A2 - Original + ArtAug": A2_WEIGHTS
}

ablation_results = {}

for experiment_name, weights_path in experiments.items():

    print(f"\nEvaluating: {experiment_name}")

    metrics, _, _ = evaluate_checkpoint(weights_path)

    ablation_results[experiment_name] = metrics

    for metric, value in metrics.items():

        print(f"{metric}: {value * 100:.2f}%")


# 10. ABLATION RESULTS TABLE

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

print("\n========== FINAL COMPARISON ==========")

print(f"{'Metric':<15} {'A1':<15} {'A2':<15}")

for metric in metric_names:

    a1 = ablation_results[
        "A1 - Original Dataset"
    ][metric] * 100

    a2 = ablation_results[
        "A2 - Original + ArtAug"
    ][metric] * 100

    print(
        f"{metric:<15} {a1:<15.2f} {a2:<15.2f}"
    )


# 11. SAVE METRICS


with open("test_metrics.txt", "w") as f:

    for experiment_name, metrics in ablation_results.items():

        f.write(f"\n{experiment_name}\n")

        for metric, value in metrics.items():

            f.write(
                f"{metric}: {value * 100:.2f}%\n"
            )

print("\nEvaluation completed successfully.")


