#!/usr/bin/env python
import os
import zipfile
import random
from shutil import copyfile
from uuid import uuid4

import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(epochs=100, id=''):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['acc'],
    )

    print(model.summary())


    TRAINING_DIR = r"data/cute-cats-n-dogs/training/"
    train_datagen = ImageDataGenerator(
          rescale=1./255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest',
    )

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=128,
        class_mode='categorical',
        target_size=(150, 150),
    )

    VALIDATION_DIR = r"data/cute-cats-n-dogs/validation/"
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        batch_size=128,
        class_mode='categorical',
        target_size=(150, 150),
    )

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        verbose=1,
        use_multiprocessing=True,
        validation_data=validation_generator,
    )

    os.makedirs(os.path.join("assets", "models", "cute-cats-n-dogs"), exist_ok=True)
    model.save(os.path.join("assets", "models", "cute-cats-n-dogs", f"{id}-model.h5"))

    return history


def plot_results(history, id=''):
    sns.set(style='darkgrid')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

    ax1.plot(epochs, acc, 'tab:blue', label='Training')
    ax1.plot(epochs, val_acc, 'tab:orange', label='Validation')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(epochs, loss, 'tab:blue', label='Training')
    ax2.plot(epochs, val_loss, 'tab:orange', label='Validation')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    os.makedirs(os.path.join("assets", "figures", "cute-cats-n-dogs"), exist_ok=True)
    plt.savefig(os.path.join("assets", "figures", "cute-cats-n-dogs", f"{id}-results.png"))

if __name__ == "__main__":
    id = uuid4()
    epochs = 5
    results = train_model(epochs, id)
    plot_results(results, id)

    results.history['epochs'] = results.epoch

    df = pd.DataFrame(results.history, index=[id] * len(results.epoch))

    if os.path.exists('assets/results/cute_cats_n_dogs.csv'):
        df.to_csv('assets/results/cute_cats_n_dogs.csv', mode='a', header=False)
    else:
        df.to_csv('assets/results/cute_cats_n_dogs.csv', mode='a', header=True)

    print(f"Session ID: {id}")
