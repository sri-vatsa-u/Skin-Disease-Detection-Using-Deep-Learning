# continue_training.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import os
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')



train_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Train"
validation_dir = r"C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\Validate"


model_save_path = r'C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\model\skin_disease_model_2.keras'
model = load_model(model_save_path)


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
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    initial_epoch=7  
)


model_save_path_new = r'C:\Users\Vatsa U\Desktop\skin disease detection\Dataset_3\model\skin_disease_model_3.keras'
model.save(model_save_path_new)
print(f"Model further trained and saved to {model_save_path_new}")


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy (continued)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss (continued)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.savefig('continued_training_validation_metrics.png')
plt.show()


#relu: introduces non linearity into the modeland learns complex patterns. output ranges from 0 to x.
#softmax: used in output layer that converts raw scores/ logits into probabilities. takes vectors of real 
# functions into probability distribution. used for classifying and providing an output that can be interpreted 
# as the model's confidence in each class.