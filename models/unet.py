import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------
# 1️⃣ Load and Preprocess Data
# ---------------------------
IMAGE_SIZE = (256, 256)  # Resize images
DATASET_PATH = "/Users/varunkarthik/Documents/mit_proj/DLRSD"  # Update with actual path

def load_images_and_masks(image_dir, mask_dir, image_size=IMAGE_SIZE):
    images, masks = [], []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        # Read image & mask
        img = cv2.imread(os.path.join(image_dir, img_file))
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Resize
        img = cv2.resize(img, image_size)
        mask = cv2.resize(mask, image_size)

        # Normalize
        img = img / 255.0  # Normalize image
        mask = mask / 255.0  # Normalize mask (values 0 or 1)
        
        # Expand dimensions for compatibility
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Paths to images and masks (Update paths)
image_dir = os.path.join(DATASET_PATH, "images")
mask_dir = os.path.join(DATASET_PATH, "masks")

# Load dataset
X, Y = load_images_and_masks(image_dir, mask_dir)

# Split data (80% Train, 20% Test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------------------------
# 2️⃣ Define U-Net Model
# ---------------------------
def unet(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = Conv2D(512, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(p4)
    c5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(u6)
    c6 = Conv2D(512, (3, 3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(u7)
    c7 = Conv2D(256, (3, 3), activation="relu", padding="same")(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(u8)
    c8 = Conv2D(128, (3, 3), activation="relu", padding="same")(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(u9)
    c9 = Conv2D(64, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs, outputs)
    return model

# Create model
model = unet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------------
# 3️⃣ Train the Model
# ---------------------------
EPOCHS = 20
BATCH_SIZE = 8

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# ---------------------------
# 4️⃣ Evaluate & Predict
# ---------------------------
# Evaluate Model
loss, acc = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {acc:.4f}")

# Predict on Test Image
idx = np.random.randint(0, len(X_test))
test_img = X_test[idx]
test_mask = Y_test[idx]

predicted_mask = model.predict(np.expand_dims(test_img, axis=0))[0]

# Plot Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(test_img)
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(test_mask[:, :, 0], cmap="gray")
plt.title("Ground Truth Mask")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask[:, :, 0], cmap="gray")
plt.title("Predicted Mask")

plt.show()
