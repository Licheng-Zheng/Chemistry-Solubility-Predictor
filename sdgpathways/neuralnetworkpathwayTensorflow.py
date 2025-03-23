import os

import keras
from sklearn.pipeline import Pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '1'



from keras.models import load_model

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

import pandas as pd
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("information.csv", delimiter=",", header=None)
dataset = dataframe.values

dropout_value = 0.0695

# This tells you how many entries there are in matrix format (15541, 121)
# print(dataset.shape)

x_data = dataset[:, :-1]
y_data = dataset[:, -1]

print(x_data)

X_training, X_validation, y_training, y_validation = train_test_split(x_data, y_data, test_size=0.75)

model = keras.Sequential([
    Input([120]),
    Dense(120, activation='relu'),
    # normalizes my data idk how to explain it properly but it makes it understandbale
    BatchNormalization(),

    # selects some nodes to remove so my model doesn't overfit (become to close to the training data)
    Dropout(dropout_value), 
    Dense(120, activation='relu'),
    BatchNormalization(), 

    Dense(120, activation='relu'),
    BatchNormalization(),
    Dropout(dropout_value),

    # Dense(120, activation='relu'),
    # BatchNormalization(),
    # Dropout(dropout_value),

    Dense(120, activation='relu'),
    BatchNormalization(),
    Dropout(dropout_value),

    Dense(120, activation='relu'),
    BatchNormalization(),
    Dropout(dropout_value),

    Dense(120, activation='sigmoid'),
    BatchNormalization(),
    # makes it so theres only one result at the end (the solubility of the compound I am predicting)
    Dense(1)
])

# some model stuff
model.compile(
    optimizer='adam',
    loss='mae'
)


print(model.summary())

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=50,
    restore_best_weights=True,
)

history = model.fit(
    X_training, y_training,
    validation_data=(X_validation, y_validation),
    batch_size=22,
    epochs=1000,
    verbose=1,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()

print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# Previous model: 1.3829

thing = input("y/n")

if thing == "y": 
    model.save('my_model.keras')  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
# model = load_model('my_model.h5')