import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model_builder import buildModel_U_net
import scipy.ndimage as ndimage
import cv2
import time
import random
import os
from pathlib import Path

IMG_WIDTH = 256
IMG_HEIGHT = 256
BATCH_SIZE = 32
EPOCHS = 100
MAX_TRAIN_IMAGES = None

TRAIN_PATH = "your/path/to/data"    
TRAIN_MASK_PATH = "your/path/to/data"  
VAL_PATH = "your/path/to/data"         
VAL_MASK_PATH = "your/path/to/data" 

# Model saving paths
MODEL_DIR = Path("./saved_models")
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "split4_oc_best_model.keras"
FINAL_MODEL_PATH = MODEL_DIR / "split4_oc_final_model.keras"
WEIGHTS_PATH = MODEL_DIR / "split4_oc_best_weights.weights.h5"

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_images_from_folder(folder_path, is_mask=False, max_images=None):
    """
    Load images from folder with optional limit on number of images
    
    Args:
        folder_path: Path to the image folder
        is_mask: Boolean indicating if loading mask images
        max_images: Maximum number of images to load (if None, load all images)
    """
    images = []
    image_files = sorted([f for f in Path(folder_path).glob('*.jpg')])
    
    # Randomly sample if max_images is specified and less than total available images
    if max_images and max_images < len(image_files):
        image_files = random.sample(image_files, max_images)
        # Sort again to maintain correspondence between images and masks
        image_files = sorted(image_files)
    
    for img_path in image_files:
        if is_mask:
            img = cv2.imread(str(img_path), 0)  # Read masks as grayscale
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.imread(str(img_path))
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        images.append(img)
    
    return np.array(images)

def step_decay(epoch):
    """Learning rate schedule"""
    step = 16
    num = epoch // step
    if num % 3 == 0:
        lrate = 1e-3
    elif num % 3 == 1:
        lrate = 1e-4
    else:
        lrate = 1e-5
    print(f'Learning rate for epoch {epoch+1} is {lrate}.')
    return float(lrate)

def preprocess_data(images, masks):
    """Preprocess images and masks"""
    # Normalize images
    processed_images = (images - np.mean(images)) / np.std(images)
    
    # Process masks
    masks = np.where(masks == 255, True, False)
    masks = 100.0 * (masks > 0)
    
    # Apply Gaussian filter to masks
    processed_masks = []
    for i in range(len(masks)):
        mask = ndimage.gaussian_filter(np.squeeze(masks[i]), sigma=(1, 1), order=0)
        processed_masks.append(mask)
    
    processed_masks = np.asarray(processed_masks, dtype='float32')
    processed_masks = np.expand_dims(processed_masks, axis=-1)
    
    return processed_images, processed_masks

def create_data_generator():
    """Create data generator for augmentation"""
    return ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    )

def train_model(train_data, train_masks, val_data, val_masks):
    """Train the model"""
    print('-'*30)
    print('Creating and compiling the FCRN model')
    print('-'*30)
    
    model = buildModel_U_net(input_dim=train_data.shape[1:])
    
    # Callbacks
    model_checkpoint = ModelCheckpoint(
        str(BEST_MODEL_PATH),  # Save best complete model
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=False
    )
    
    weights_checkpoint = ModelCheckpoint(
        str(WEIGHTS_PATH),  # Save best weights separately
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True
    )
    
    lr_scheduler = LearningRateScheduler(step_decay)
    
    # Data generator
    datagen = create_data_generator()
    
    print('Fitting model...')
    print('-'*30)
    
    # Fit model
    history = model.fit(
        datagen.flow(
            train_data,
            train_masks,
            batch_size=BATCH_SIZE
        ),
        steps_per_epoch=len(train_data),
        epochs=EPOCHS,
        callbacks=[model_checkpoint, weights_checkpoint, lr_scheduler],
        validation_data=(val_data, val_masks)
    )
    
    # Save final model
    model.save(str(FINAL_MODEL_PATH))
    
    # Save training history
    np.save(str(MODEL_DIR / 'training_history.npy'), history.history)
    
    return model

def detect(model, test_data, threshold=0.57):
    """Perform detection on test data"""
    # Load the best model for predictions
    if BEST_MODEL_PATH.exists():
        model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        print("Loaded best model for predictions")
    else:
        print("Using final model for predictions")
    
    start = time.time()
    predictions = model.predict(test_data)
    print(f"\nConsumed time: {time.time()-start:.2f} s\n")
    
    preds_test = np.where(predictions > 0, predictions / 100, predictions)
    preds_test = (preds_test + 1) / 2
    preds_test_t = (preds_test > threshold).astype(np.uint8)
    
    return preds_test_t, preds_test

def save_results(predictions, binary_predictions, val_image_paths):
    """Save prediction results"""
    results_dir = Path("./PredMasks")
    results_dir.mkdir(exist_ok=True)
    
    for i, img_path in enumerate(val_image_paths):
        img_name = img_path.stem
        
        # Save binary mask
        pred_mask = cv2.resize(
            np.squeeze(binary_predictions)[i],
            (6496, 3360),
            interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(
            str(results_dir / f"{img_name}_pred_mask.jpg"),
            pred_mask * 255,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )
        
        # Save probability map
        pmap = cv2.resize(
            predictions[i, :, :, 0],
            (6496, 3360),
            interpolation=cv2.INTER_CUBIC
        )
        pmap_path = str(results_dir / f"{img_name}_PMap.jpg")
        cv2.imwrite(
            pmap_path,
            pmap * 255,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        )

def main():
    print(f"Model and weights will be saved in: {MODEL_DIR}")
    
    # Load training data with size limit
    train_images = load_images_from_folder(TRAIN_PATH, max_images=MAX_TRAIN_IMAGES)
    train_masks = load_images_from_folder(TRAIN_MASK_PATH, is_mask=True, max_images=MAX_TRAIN_IMAGES)
    
    print(f"Training with {len(train_images)} images")
    
    # Load validation data (keeping full validation set)
    val_images = load_images_from_folder(VAL_PATH)
    val_masks = load_images_from_folder(VAL_MASK_PATH, is_mask=True)
    
    # Preprocess training data
    train_data, train_masks = preprocess_data(train_images, train_masks)
    
    # Preprocess validation data
    val_data, val_masks = preprocess_data(val_images, val_masks)
    
    # Train model
    model = train_model(train_data, train_masks, val_data, val_masks)
    
    # Perform detection on validation set
    binary_predictions, predictions = detect(model, val_data, threshold=0.60)
    
    # Save results
    print("Saving results...")
    val_image_paths = sorted(Path(VAL_PATH).glob('*.jpg'))
    save_results(predictions, binary_predictions, val_image_paths)
    
    print("Training and prediction completed successfully!")
    print(f"Model saved in: {MODEL_DIR}")

if __name__ == "__main__":
    main()