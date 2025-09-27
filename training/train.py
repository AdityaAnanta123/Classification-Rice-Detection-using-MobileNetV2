# training/train.py
"""
Train pipeline:
1. Load datasets via preprocessing.create_datasets_from_directory
2. Build model via model.build_mobilenetv2_model
3. Train with callbacks (ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger)
4. Save trained model (.keras) + plots + class indices
"""

import os
from pathlib import Path
import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from preprocessing.preprocessed import (
    create_datasets_from_directory,
    augmentation_layer,
    save_class_indices
)
from model.mobilenet_model import build_mobilenetv2_model
from training.utils import plot_history, save_class_map

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
CLASS_MAP_PATH = ARTIFACTS_DIR / "class_map.json"


def train(data_dir: str,
          image_size=(224, 224),
          batch_size: int = 16,
          epochs: int = 10,
          learning_rate: float = 1e-4,
          fine_tune_at: int = 100,
          dropout_rate: float = 0.4):

    # create dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # load datasets
    train_ds, val_ds, test_ds, class_indices = create_datasets_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size
    )

    # add cache/shuffle/prefetch
    train_ds = train_ds.shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    num_classes = len(class_indices)
    print("Num classes:", num_classes)

    # augmentation only for training
    aug = augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # build model
    model = build_mobilenetv2_model(
        num_classes=num_classes,
        input_shape=(*image_size, 3),
        base_trainable=False,
        fine_tune_at=fine_tune_at,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    model.summary()

    # callbacks
    checkpoint_path = MODELS_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(str(PLOTS_DIR / "training_log.csv"))
    ]

    # training
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Simpan mapping kelas
    Path("artifacts/class_indices").mkdir(parents=True, exist_ok=True)
    save_class_indices(class_indices, "artifacts/class_indices/class_indices.json")
    
    # save final model
    final_model_path = MODELS_DIR / "final_model.keras"
    model.save(final_model_path, include_optimizer=False)

    # plot training history
    plot_history(history, PLOTS_DIR / "training_history.png")

    # save class map
    save_class_map(class_indices, CLASS_MAP_PATH)

    # evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    

    return {
        "model_path": str(final_model_path),
        "checkpoint": str(checkpoint_path),
        "plot_history": str(PLOTS_DIR / "training_history.png"),
        "class_map": str(CLASS_MAP_PATH)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default="data/raw/Rice_Image_Dataset",
                        help="Path to dataset root (folders per class).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--fine_tune_at", type=int, default=100)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        fine_tune_at=args.fine_tune_at
    )
