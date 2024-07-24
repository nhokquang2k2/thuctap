
import matplotlib
matplotlib.use('Agg')

from livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True,
                    help='Path to input Dataset')
parser.add_argument('-m', '--model', type=str, required=True,
                    help='Path to output trained model')
parser.add_argument('-p', '--plot', type=str, default='plot.png',
                    help='Path to output loss/accuracy plot')
args = vars(parser.parse_args())

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))
data = list()
labels = list()

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype='float') / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = tf.keras.utils.to_categorical(labels, 2)  # one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                      zoom_range=0.15,
                                                      width_shift_range=0.2,
                                                      height_shift_range=0.2,
                                                      shear_range=0.15,
                                                      horizontal_flip=True,
                                                      fill_mode='nearest')

INIT_LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 50

print('[INFO] compiling model...')
optimizer = tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3, classes=len(le.classes_))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

print(f'[INFO] training model for {EPOCHS} epochs...')
history = model.fit(x=aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    validation_data=(X_test, y_test),
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    epochs=EPOCHS)

print('[INFO] evaluating network...')
predictions = model.predict(x=X_test, batch_size=BATCH_SIZE)

print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print(f"[INFO serializing network to '{args['model']}'")
model.save(args['model'], save_format='h5')
model.export("model/1")
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, EPOCHS), history.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), history.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), history.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), history.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])
