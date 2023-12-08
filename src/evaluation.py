import torch
from torch import nn
from torch.utils.data import DataLoader
import seaborn as sns
from src.configuration import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model: nn.Module, model_path: str, test_loader: DataLoader, device: torch.device):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    model.eval()
    label_true, label_pred = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = torch.max(model(data), dim=1)[1]
            label_true.extend(label.tolist())
            label_pred.extend(pred.tolist())
    
    accuracy = accuracy_score(label_true, label_pred)
    precision = precision_score(label_true, label_pred, average="macro")
    recall = recall_score(label_true, label_pred, average="macro")
    f1 = f1_score(label_true, label_pred, average="macro")
    
    print("----------------------------------------")
    print("Model at %s:" % model_path)
    print("----------------------------------------")
    print("Accuracy: %.4f" % accuracy)
    print("Precision: %.4f" % precision)
    print("Recall: %.4f" % recall)
    print("F1 Score: %.4f" % f1)
    
    cm = confusion_matrix(label_true, label_pred)
    sns.heatmap(cm, annot=True, xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, cmap="YlGn")