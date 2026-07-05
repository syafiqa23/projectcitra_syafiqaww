import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

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

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    base_dataset = datasets.ImageFolder(DATASET_DIR)
    class_names = base_dataset.classes

    idx = list(range(len(base_dataset)))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=SEED,
        stratify=base_dataset.targets
    )

    train_dataset = Subset(
        datasets.ImageFolder(DATASET_DIR, transform=train_transform),
        train_idx
    )
    val_dataset = Subset(
        datasets.ImageFolder(DATASET_DIR, transform=val_transform),
        val_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0

    def train_epoch():
        model.train()
        tl, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        return tl/len(train_loader), correct/total

    def validate():
        model.eval()
        vl, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                vl += loss.item()
                pred = outputs.argmax(1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return vl/len(val_loader), correct/total

    print("STAGE 1")
    for e in range(STAGE1_EPOCHS):
        trl, tra = train_epoch()
        val_l, val_a = validate()
        train_losses.append(trl); val_losses.append(val_l)
        train_accs.append(tra); val_accs.append(val_a)
        print(f"Epoch {e+1}/{STAGE1_EPOCHS} TrainAcc={tra:.4f} ValAcc={val_a:.4f}")
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), f"{MODEL_DIR}/mobilenetv2_best.pth")

    print("FINE TUNING")
    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for e in range(FINETUNE_EPOCHS):
        trl, tra = train_epoch()
        val_l, val_a = validate()
        train_losses.append(trl); val_losses.append(val_l)
        train_accs.append(tra); val_accs.append(val_a)
        print(f"FineTune {e+1}/{FINETUNE_EPOCHS} TrainAcc={tra:.4f} ValAcc={val_a:.4f}")
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), f"{MODEL_DIR}/mobilenetv2_best.pth")

    model.load_state_dict(torch.load(f"{MODEL_DIR}/mobilenetv2_best.pth", map_location=device))
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(f"{RESULT_DIR}/mobilenet_classification_report.txt", "w") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names)
    plt.savefig(f"{RESULT_DIR}/mobilenet_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_accs); plt.plot(val_accs)
    plt.legend(["Train","Val"]); plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(train_losses); plt.plot(val_losses)
    plt.legend(["Train","Val"]); plt.title("Loss")
    plt.savefig(f"{RESULT_DIR}/mobilenet_training_curve.png", dpi=300)
    plt.close()

    confidence_scores = np.max(np.array(all_probs), axis=1)

    plt.figure()
    plt.hist(confidence_scores, bins=20)
    plt.savefig(f"{RESULT_DIR}/mobilenet_confidence_distribution.png", dpi=300)
    plt.close()

    p33 = np.percentile(confidence_scores, 33)
    p66 = np.percentile(confidence_scores, 66)

    degradation = []
    for s in confidence_scores:
        if s < p33:
            degradation.append("High Degradation")
        elif s < p66:
            degradation.append("Medium Degradation")
        else:
            degradation.append("Low Degradation")

    print(Counter(degradation))

    pd.Series(degradation).value_counts().plot(kind="bar")
    plt.title("Adaptive Peatland Degradation")
    plt.savefig(f"{RESULT_DIR}/mobilenet_degradation_distribution.png", dpi=300)
    plt.close()

    print("Best Validation Accuracy:", best_acc)

if __name__ == "__main__":
    main()
