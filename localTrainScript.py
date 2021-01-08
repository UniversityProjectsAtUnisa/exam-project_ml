import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import h5py
import numpy as np
import keras


# Dataset Loading
file_h5 = 'data/train.h5'
f = h5py.File(file_h5, 'r')
X = f['X'][...]
y = f['y'][...]

# Model
model = keras.models.Sequential()

model.add(keras.layers.Reshape((1000, 64, 1), input_shape=(1000, 64)))
model.add(keras.layers.Conv2D(128, 5, padding='same', activation='relu'))
model.add(keras.layers.Conv2D(128, 5, padding='same', activation='relu'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(55, activation='softmax'))

model.summary()

model.compile(
    # optimizer=keras.optimizers.RMSprop(learning_rate=0.03),
    optimizer=keras.optimizers.RMSprop(lr=0.03),
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(X, y, batch_size=32, epochs=10)

model.save('./models/local_model.h5')
