import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras._tf_keras.keras.utils import to_categorical, plot_model
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Input
from keras._tf_keras.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob

# Initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

# Load and preprocess data
def load_and_preprocess_data():
    data = []
    labels = []
    image_files = [f for f in glob.glob(r'./gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
    random.shuffle(image_files)

    for img in image_files:
        image = cv2.imread(img)
        image = cv2.resize(image, (img_dims[0], img_dims[1]))
        data.append(image)
        
        label = img.split(os.path.sep)[-2]
        label = 1 if label == "woman" else 0
        labels.append(label)

    return np.array(data, dtype="float32") / 255.0, np.array(labels)

data, labels = load_and_preprocess_data()

# Split dataset
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = tf.keras.utils.to_categorical(trainY, num_classes=2)
testY = tf.keras.utils.to_categorical(testY, num_classes=2)

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, label

# Create tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = (train_dataset
                 .shuffle(buffer_size=len(trainX))
                 .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                 .batch(batch_size)
                 .prefetch(tf.data.AUTOTUNE))

test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY)).batch(batch_size)

# Define model architecture
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model.add(Input(shape=inputShape))
    model.add(Conv2D(32, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build and compile model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    verbose=1
)

# Save model
model.save('./machine_learning/models/gender_detection')

# Plot training history
plt.style.use("ggplot")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()