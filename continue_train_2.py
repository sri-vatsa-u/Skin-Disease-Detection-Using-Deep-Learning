# continue_training_with_finetune.py

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import os
import sys
import io
import matplotlib.pyplot as plt

# Ensure stdout uses UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define directories
train_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Train"
validation_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Validate"

# Use a pre-trained EfficientNetB0 model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model to start with fine-tuning

# Build a new model on top
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(os.listdir(train_dir)), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ImageDataGenerators (same as before, but increase augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  # Increased zoom
    brightness_range=[0.8, 1.2],  # Added brightness variation
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7)

def lr_schedule(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Start with a few epochs for initial fine-tuning
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
)

# Unfreeze the base model for full fine-tuning
base_model.trainable = True

# Re-compile with a lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Continue training for fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=30,  # Fine-tune for longer
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    initial_epoch=10  # Continue from previous training
)

# Save the fine-tuned model
model_save_path_new = r'C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\model\skin_disease_model_finetuned.keras'
model.save(model_save_path_new)
print(f"Fine-tuned model saved to {model_save_path_new}")

# Plot the training and validation metrics
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_finetune.history['accuracy'])
plt.plot(history_finetune.history['val_accuracy'])
plt.title('Model accuracy (fine-tuned)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_finetune.history['loss'])
plt.plot(history_finetune.history['val_loss'])
plt.title('Model loss (fine-tuned)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.savefig('fine_tuned_training_validation_metrics.png')
plt.show()
