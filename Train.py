import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling, RandomContrast
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

plt.style.use('seaborn-dark')

train_data = image_dataset_from_directory(r'/Users/soumallyadey/Desktop/project/MaskDetection/train', labels='inferred',
                                          label_mode='binary', interpolation='nearest', image_size=[128, 128],
                                          batch_size=32, shuffle=True)

class_name = train_data.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    Rescaling(1.0 / 255, input_shape=(128, 128, 3)),
    RandomFlip(),
    RandomRotation(0.4),
    RandomContrast(0.3),

    # First Layer
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Second Layer
    layers.Conv2D(filters=128, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Third Layer
    layers.Conv2D(filters=256, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Forth Layer
    layers.Conv2D(filters=256, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Fifth Layer
    layers.Conv2D(filters=256, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Dropout(0.2),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(units=6, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')

])

model.compile(optimizer=tf.optimizers.Adam(epsilon=0.02), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=train_data, epochs=20, batch_size=32)

model.save('face_detection.h5')

print(model.metrics_names)
history_dir = history.history
print(history_dir)
hist_df = pd.DataFrame(history.history)
hist_df.loc[:, ['loss', 'val_loss']].plot()
hist_df.loc[:, ['accuracy', 'val_accuracy']].plot()
N = 50
plt.style.use("ggplot")
plt.figure()
plt.plot(history_dir["loss"], label="train_loss")
plt.plot(history_dir["accuracy"], label="train_acc")
plt.plot(history_dir["val_loss"], label="val_loss")
plt.plot(history_dir["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
model = tf.keras.models.load_model('face_detection.h5')
