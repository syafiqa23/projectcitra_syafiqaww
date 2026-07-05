import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIG
# =====================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 16

STAGE1_EPOCHS = 15
FINETUNE_EPOCHS = 10

NUM_CLASSES = 3

DATASET_DIR = "dataset"
MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# MAIN
# =====================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("DEVICE :", device)

    if torch.cuda.is_available():
        print(
            "GPU :",
            torch.cuda.get_device_name(0)
        )

    print("=" * 60)

    # =====================================================
    # TRANSFORM
    # =====================================================

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomRotation(20),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # =====================================================
    # DATASET
    # =====================================================

    base_dataset = datasets.ImageFolder(
        DATASET_DIR
    )

    class_names = base_dataset.classes

    print("Classes :", class_names)

    indices = list(
        range(len(base_dataset))
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=base_dataset.targets
    )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=val_transform
    )

    train_dataset = Subset(
        train_dataset,
        train_idx
    )

    val_dataset = Subset(
        val_dataset,
        val_idx
    )

    print("Train :", len(train_dataset))
    print("Val :", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # =====================================================
    # MODEL
    # =====================================================

    weights = models.ResNet101_Weights.DEFAULT

    model = models.resnet101(
        weights=weights
    )

    # freeze backbone

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(

        nn.Dropout(0.5),

        nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )
    )

    model = model.to(device)

    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    # =====================================================
    # TRAIN FUNCTION
    # =====================================================

    def train_epoch():

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        return (
            running_loss / len(train_loader),
            correct / total
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate():

        model.eval()

        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                running_loss += loss.item()

                _, predicted = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        return (
            running_loss / len(val_loader),
            correct / total
        )

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    best_acc = 0

    # =====================================================
    # STAGE 1
    # =====================================================

    print("\nSTAGE 1 TRAINING\n")

    for epoch in range(STAGE1_EPOCHS):

        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch+1}/{STAGE1_EPOCHS}"
            f" | Train Acc {train_acc:.4f}"
            f" | Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/resnet101_best.pth"
            )

    # =====================================================
    # FINE TUNING
    # =====================================================

    print("\nFINE TUNING\n")

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-5
    )

    for epoch in range(FINETUNE_EPOCHS):

        train_loss, train_acc = train_epoch()
        val_loss, val_acc = validate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"FineTune {epoch+1}/{FINETUNE_EPOCHS}"
            f" | Train Acc {train_acc:.4f}"
            f" | Val Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/resnet101_best.pth"
            )

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    model.load_state_dict(
        torch.load(
            f"{MODEL_DIR}/resnet101_best.pth",
            map_location=device
        )
    )

    model.eval()

    # =====================================================
    # EVALUATION
    # =====================================================

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    print(report)

    with open(
        f"{RESULT_DIR}/resnet101_classification_report.txt",
        "w"
    ) as f:
        f.write(report)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("ResNet101 Confusion Matrix")

    plt.savefig(
        f"{RESULT_DIR}/resnet101_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # TRAINING CURVE
    # =====================================================

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title("Accuracy")
    plt.legend(["Train","Val"])

    plt.subplot(1,2,2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Loss")
    plt.legend(["Train","Val"])

    plt.savefig(
        f"{RESULT_DIR}/resnet101_training_curve.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence_scores = np.max(
        np.array(all_probs),
        axis=1
    )

    plt.figure(figsize=(8,5))

    plt.hist(
        confidence_scores,
        bins=20
    )

    plt.title(
        "Confidence Distribution"
    )

    plt.savefig(
        f"{RESULT_DIR}/resnet101_confidence_distribution.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # DEGRADATION
    # =====================================================

    p33 = np.percentile(
        confidence_scores,
        33
    )

    p66 = np.percentile(
        confidence_scores,
        66
    )

    degradation = []

    for score in confidence_scores:

        if score < p33:
            degradation.append(
                "High Degradation"
            )

        elif score < p66:
            degradation.append(
                "Medium Degradation"
            )

        else:
            degradation.append(
                "Low Degradation"
            )

    print(
        Counter(degradation)
    )

    pd.Series(
        degradation
    ).value_counts().plot(
        kind="bar"
    )

    plt.title(
        "Adaptive Peatland Degradation"
    )

    plt.savefig(
        f"{RESULT_DIR}/resnet101_degradation_distribution.png",
        dpi=300
    )

    plt.close()

    print("\nDONE")
    print(
        "Best Accuracy :",
        best_acc
    )

if __name__ == "__main__":
    main()