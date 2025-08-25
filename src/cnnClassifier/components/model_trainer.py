import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import math

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
        versioned_path = base_path.parent / f"model_{timestamp}.keras"
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

        # self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        # self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        self.steps_per_epoch = math.ceil(self.train_generator.samples / self.train_generator.batch_size)
        self.validation_steps = math.ceil(self.valid_generator.samples / self.valid_generator.batch_size)

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        # ✅ 1. Save latest model (overwrite)
        self.model.save(self.config.trained_model_path)
        print(f" Latest model saved at: {self.config.trained_model_path}")

        # ✅ 2. Save versioned model (timestamp)
        versioned_model_path = self.save_model(
            base_path=self.config.trained_model_path,
            model=self.model
        )

        return history, versioned_model_path

# import os
# import math
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.metrics import confusion_matrix, classification_report


# class ModelTrainer:
#     def __init__(self, data_dir, img_size=(150, 150), batch_size=32, epochs=10):
#         self.data_dir = data_dir
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.model = None

#     def build_model(self):
#         """Builds a simple CNN model"""
#         self.model = Sequential([
#             Conv2D(32, (3, 3), activation="relu", input_shape=(self.img_size[0], self.img_size[1], 3)),
#             MaxPooling2D(2, 2),
#             Conv2D(64, (3, 3), activation="relu"),
#             MaxPooling2D(2, 2),
#             Flatten(),
#             Dense(128, activation="relu"),
#             Dropout(0.5),
#             Dense(2, activation="softmax")  # 2 classes
#         ])
#         self.model.compile(optimizer=Adam(learning_rate=0.001),
#                            loss="categorical_crossentropy",
#                            metrics=["accuracy"])

#     def prepare_data(self):
#         """Prepares training and validation datasets with augmentation"""
#         datagen = ImageDataGenerator(
#             rescale=1.0 / 255,
#             shear_range=0.2,
#             zoom_range=0.2,
#             horizontal_flip=True,
#             rotation_range=15,
#             validation_split=0.2
#         )

#         self.train_generator = datagen.flow_from_directory(
#             self.data_dir,
#             target_size=self.img_size,
#             batch_size=self.batch_size,
#             class_mode="categorical",
#             subset="training"
#         )

#         self.valid_generator = datagen.flow_from_directory(
#             self.data_dir,
#             target_size=self.img_size,
#             batch_size=self.batch_size,
#             class_mode="categorical",
#             subset="validation",
#             shuffle=False
#         )

#         # ✅ Fix: use ceil instead of // to avoid "ran out of data"
#         self.steps_per_epoch = math.ceil(self.train_generator.samples / self.batch_size)
#         self.validation_steps = math.ceil(self.valid_generator.samples / self.batch_size)

#     def train(self):
#         """Trains the model"""
#         callbacks = [
#             EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
#             ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
#         ]

#         history = self.model.fit(
#             self.train_generator,
#             steps_per_epoch=self.steps_per_epoch,
#             epochs=self.epochs,
#             validation_data=self.valid_generator,
#             validation_steps=self.validation_steps,
#             callbacks=callbacks
#         )

#         # Save model in modern format
#         self.model.save("chest_xray_model.keras")
#         print("✅ Model saved as chest_xray_model.keras")

#         return history

#     def evaluate_confusion_matrix(self):
#         """Evaluates model on validation set with confusion matrix"""
#         y_true = []
#         y_pred = []

#         # Go through validation generator once
#         for images, labels in self.valid_generator:
#             preds = self.model.predict(images, verbose=0)
#             y_pred.extend(np.argmax(preds, axis=1))
#             y_true.extend(np.argmax(labels, axis=1))
#             if len(y_true) >= self.valid_generator.samples:
#                 break

#         cm = confusion_matrix(y_true, y_pred)
#         cr = classification_report(y_true, y_pred, target_names=list(self.valid_generator.class_indices.keys()))

#         print("\n🔹 Confusion Matrix:\n", cm)
#         print("\n🔹 Classification Report:\n", cr)


# if __name__ == "__main__":
#     data_dir = "artifacts\data_ingestion\dataset_mlops"  # Change this if your dataset path is different
#     trainer = ModelTrainer(data_dir)
#     trainer.build_model()
#     trainer.prepare_data()
#     trainer.train()
#     trainer.evaluate_confusion_matrix()




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