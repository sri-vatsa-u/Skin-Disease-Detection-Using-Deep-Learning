# train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import os
import sys
import io
import matplotlib.pyplot as plt


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


train_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Train"
validation_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Validate"


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

def lr_schedule(epoch, lr):
    
    if epoch % 10 == 0 and epoch != 0:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)


history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr, lr_scheduler]
)


model_save_path = r'C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\model\skin_disease_model_1.keras'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.savefig('training_validation_metrics.png')
plt.show()



# Define the path to the original dataset
#original_dataset_dir = r'D:\MyNotes\Project\Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis\IMG_CLASSES'  # Replace with the path to your original dataset directory
#train_dir = r'D:\MyNotes\Project\Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis\Training'  # Replace with the path where you want to save the training data
#validation_dir = r'D:\MyNotes\Project\Skin-Disease-Image-Classifier-for-Accurate-and-Accessible-Diagnosis\Validation'  # Replace with the path where you want to save the validation data

#adam preferred since it takes the advantages of adagrad and rmsprop like building momentum and maintaining high learning rates
#catagorical_crossentropy compares predicted probabilities to true class labels. preferred for exclusive multiclass unlike 
# binary and sparse(which compares to integer values instead of class labels)


