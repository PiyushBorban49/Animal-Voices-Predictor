import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from xgboost import XGBRegressor, XGBClassifier

from others.testWorld import activation1

data = []
labels = []
animals = ['Bear','Cat','Chicken','Cow','Dog','Dolphin','Donkey','Elephant','Frog','Horse','Lion','Monkey','Sheep']

base_animal_sound = r"C:\Users\piyus\OneDrive\Desktop\python\artificial intelligence\Animal Help Call\Animal-Soundprepros"
n_mfcc = 20
max_len = 100

for i, animal in enumerate(animals):
    for j in range(1,50):
        animal_sound = os.path.join(base_animal_sound,animal,f"{animal}_{j}.wav")
        y, sr = librosa.load(animal_sound,duration=10.0)
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        data.append(mfcc.flatten())
        labels.append(i)

X = np.array(data)
Y = np.array(labels)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# clf = XGBClassifier(
#     n_estimators=100,
#     max_depth=6,
#     learning_rate=0.1,
#     random_state=42
# )
# clf.fit(X_train,Y_train)
# print(clf.score(X_test,Y_test))


model = Sequential()

model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],),
                                kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=regularizers.l2(0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(len(animals), activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

callbacks = []

# 1. Early Stopping - stops training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',           # Metric to monitor
    patience=10,                  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,    # Restore model weights from the best epoch
    verbose=1
)
callbacks.append(early_stopping)

history = model.fit(X_train,Y_train,verbose=1,validation_data=(X_test,Y_test),epochs=50,batch_size=32,callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()