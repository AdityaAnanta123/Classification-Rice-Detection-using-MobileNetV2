# training/utils.py
import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_history(history, out_path):
    """
    Plot training & validation loss/accuracy and save to out_path
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10,4))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.legend()
    plt.title("Accuracy")

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_class_map(class_map: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(class_map, f, indent=2)

def load_class_map(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
