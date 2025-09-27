import tensorflow as tf #for preprocessing image using deep learning
import os #for navigation and manajement file / folder
import json # for save and load file type .json
import random #for randomize
from pathlib import Path #for navigation and manajement file / folder
from typing import Tuple, Dict #for type data in parameter function

AUTOTUNE = tf.data.AUTOTUNE


#Load and Preprocess Images
def load_and_preprocess_image(path: str, label: int, image_size=(224, 224)):
    """
    Load image, decode JPEG, resize, normalize to [0,1].
    """
    image = tf.io.read_file(path) #read file image from directory raw
    image = tf.image.decode_jpeg(image, channels=3)  #decode image using decode_jpeg
    image = tf.image.resize(image, image_size) #resize image to image size (224,224)
    image.set_shape([image_size[0], image_size[1], 3])  # fix error shape 
    image = tf.cast(image, tf.float32) / 255.0 #normalize image to [0,1]
    return image, label


#Augmentation Layers (Change input image at random every time the image is training)
def augmentation_layer():
    """
    Return keras Sequential for data augmentation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), #flip data randomize horizontal
        tf.keras.layers.RandomRotation(0.08), #change data based on rotation with max corner 0.08
        tf.keras.layers.RandomZoom(0.08), #change data based on zoom with max zoom 0.08
        tf.keras.layers.RandomHeight(0.06), #change data based on height with max randomize 0.06
        tf.keras.layers.RandomWidth(0.06), #change data based on width with max randomize 0.06
    ], name="data_augmentation")

#create dataset from raw directory
def create_datasets_from_directory(data_dir: str, #for path dataset directory
                                   image_size: Tuple[int,int]=(224,224), #for image size in image 
                                   batch_size: int=32, #batch size for processing dataset
                                   validation_split: float=0.2, #validation split for dataset 
                                   test_split: float=0.1, #test split for dataset
                                   seed: int=123) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Split data ke train/val/test dataset.
    """
    data_dir = Path(data_dir) # made a variable for directory data
    if not data_dir.exists(): #checking if directory data is existing or not
        raise FileNotFoundError(f"{data_dir} not found.")

    # take all label and file 
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()]) #sorting class label using ascending
    all_file_paths, all_labels = [], []
    for label_index, class_name in enumerate(class_names): #take all file in raw directory in .jpg, .jpeg, and .png format
        class_dir = data_dir / class_name
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in class_dir.glob(ext):
                all_file_paths.append(str(p))
                all_labels.append(label_index)

    if len(all_file_paths) == 0: #checking if file doesn't have any images
        raise ValueError("No images found in dataset directory!")

    # shuffle data so model can learn from unordered data but still paired
    combined = list(zip(all_file_paths, all_labels))
    random.Random(seed).shuffle(combined)
    all_file_paths, all_labels = zip(*combined)

    # made a splitting data from total data to test, val, and train data
    n_total = len(all_file_paths)
    n_test = int(n_total * test_split)
    n_val = int((n_total - n_test) * validation_split)
    n_train = n_total - n_test - n_val

    train_paths = all_file_paths[:n_train]
    train_labels = all_labels[:n_train]
    val_paths = all_file_paths[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]
    test_paths = all_file_paths[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]

    #made a path to dataset that already be preprocessing
    def paths_to_dataset(paths, labels, shuffle=False):
        paths_ds = tf.data.Dataset.from_tensor_slices(list(paths))
        labels_ds = tf.data.Dataset.from_tensor_slices(list(labels))
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))
        ds = ds.map(lambda p, l: load_and_preprocess_image(p, l, image_size),
                    num_parallel_calls=AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=1000, seed=seed)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    #insert data to directory train, val, and test
    train_ds = paths_to_dataset(train_paths, train_labels, shuffle=True)
    val_ds = paths_to_dataset(val_paths, val_labels)
    test_ds = paths_to_dataset(test_paths, test_labels)

    #get back dataset that already used and information about index class
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    return train_ds, val_ds, test_ds, class_indices

#save class indices to class_indices directory with .json format
def save_class_indices(mapping: dict, filepath: str):
    """
    Simpan mapping class->index ke JSON.
    """
    with open(filepath, "w") as f:
        json.dump(mapping, f, indent=2)

#save processed images to processed directory
def save_processed_images(dataset, output_dir: Path, class_indices: Dict, limit: int = 5):
    """
    Simpan hasil preprocessing ke folder processed/ (preview).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    inv_class_indices = {v: k for k, v in class_indices.items()}

    saved = 0
    for images, labels in dataset.take(limit):  #just need several batch
        for i in range(images.shape[0]):
            img = tf.keras.utils.array_to_img(images[i].numpy())
            label = labels[i].numpy()
            class_name = inv_class_indices[label]

            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img_path = class_dir / f"processed_{saved}.jpg"
            img.save(img_path)
            saved += 1
    print(f"✅ Saved {saved} processed images to {output_dir}")

#main for information about dataset and directory processed
if __name__ == "__main__":
    DATASET_DIR = "./data/raw/Rice_Image_Dataset"
    PROCESSED_DIR = "./data/processed"

    train_ds, val_ds, test_ds, class_indices = create_datasets_from_directory(DATASET_DIR)

    save_class_indices(class_indices, os.path.join(PROCESSED_DIR, "class_indices.json"))

    # save some examples of train, val, and test data
    save_processed_images(train_ds, Path(PROCESSED_DIR) / "train", class_indices, limit=5)
    save_processed_images(val_ds, Path(PROCESSED_DIR) / "val", class_indices, limit=2)
    save_processed_images(test_ds, Path(PROCESSED_DIR) / "test", class_indices, limit=2)
