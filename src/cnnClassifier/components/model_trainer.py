import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from datetime import datetime


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        """Load the updated base model"""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    

    # def train_valid_generator(self):
    #     """Prepare training and validation data generators"""

    #     datagenerator_kwargs = dict(
    #         rescale=1.0 / 255,
    #         validation_split=0.20
    #     )

    #     dataflow_kwargs = dict(
    #         target_size=self.config.params_image_size[:-1],
    #         batch_size=self.config.params_batch_size,
    #         interpolation="bilinear"
    #     )

    #     # Validation generator
    #     valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #         **datagenerator_kwargs
    #     )

    #     self.valid_generator = valid_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="validation",
    #         shuffle=False,
    #         class_mode="sparse",  # <-- integer labels for sparse loss
    #         **dataflow_kwargs
    #     )

    #     # Training generator
    #     if self.config.params_is_augmentation:
    #         train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    #             rotation_range=40,
    #             horizontal_flip=True,
    #             width_shift_range=0.2,
    #             height_shift_range=0.2,
    #             shear_range=0.2,
    #             zoom_range=0.2,
    #             **datagenerator_kwargs
    #         )
    #     else:
    #         train_datagenerator = valid_datagenerator

    #     self.train_generator = train_datagenerator.flow_from_directory(
    #         directory=self.config.training_data,
    #         subset="training",
    #         shuffle=True,
    #         class_mode="sparse",  # <-- integer labels for sparse loss
    #         **dataflow_kwargs
    #     )
    def train_valid_generator(self):
        """Prepare training and validation data generators"""

        # Common generator kwargs
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation generator (integer labels)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            class_mode="sparse",  # <-- integer labels
            **dataflow_kwargs
        )

        # Training generator
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="sparse",  # <-- integer labels
            **dataflow_kwargs
        )

        # ✅ Debug print
        print("Class indices mapping:", self.train_generator.class_indices)
        x_batch, y_batch = next(self.train_generator)
        print("Sample labels batch (integers):", y_batch[:10])


    @staticmethod
    def save_model(base_path: Path, model: tf.keras.Model):
        """Save trained model with timestamp (versioning)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = base_path.parent / f"model_{timestamp}.h5"
        model.save(versioned_path)
        print(f"✅ Model saved at: {versioned_path}")
        return versioned_path

    def train(self):
        """Compile and train the model"""

        # ✅ Compile
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.params_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # ✅ 1. Save latest model (overwrite)
        self.model.save(self.config.trained_model_path)
        print(f"✅ Latest model saved at: {self.config.trained_model_path}")

        # ✅ 2. Save versioned model (timestamp)
        versioned_model_path = self.save_model(
            base_path=self.config.trained_model_path,
            model=self.model
        )

        return history, versioned_model_path

