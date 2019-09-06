import os
import zipfile
import random
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def main():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['acc'],
    )

    print(model.summary())


    TRAINING_DIR = r"data/"
    train_datagen = ImageDataGenerator(rescale=1/255.)
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        batch_size=128,
        class_mode='binary',
        target_size=(150, 150),
    )

    history = model.fit_generator(
        train_generator,
        epochs=10,
        verbose=1,
        use_multiprocessing=True,
        # validation_data=validation_generator,
    )


if __name__ == "__main__":
    main()
