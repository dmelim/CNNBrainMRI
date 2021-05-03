import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells

#Importing data using DataGenerator and flow

datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.3,
    dtype=tf.float32,
)


# Load training and validation sets
ds_train = datagen.flow_from_directory(
    '../input/brainmriv2/brain_tumor_dataset',
    target_size=[128, 128],
    batch_size=64,
    color_mode = 'grayscale',
    shuffle=True,
    subset = 'training',
)
ds_valid = datagen.flow_from_directory(
    '../input/brainmriv2/brain_tumor_dataset',
    target_size=[128, 128],
    batch_size=25,
    color_mode = 'grayscale',
    shuffle=True,
    subset = 'validation',
)

#Model 
model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
                  input_shape=[128, 128, 1]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax'),
])

#Training the Model
early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=25,
    restore_best_weights=True,
)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    callbacks=[early_stopping],
    epochs=100,
    verbose=0,
)

#Seeing the results

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot();
print(history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy', 'loss', 'val_loss']])
