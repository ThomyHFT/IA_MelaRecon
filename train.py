import os
import tensorflow as tf
from src.data_loader import get_data_generators
from src.model_builder import build_model
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data/processed'
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = 'exported_models/best_model.h5'

# Create folder if not exists 
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# load generators
train_gen, val_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

# build model
model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
]

# train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# save history of train
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

plt.savefig('exported_models/training_history.png')
plt.close()
