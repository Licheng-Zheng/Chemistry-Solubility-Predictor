import os
import numpy as np
from mefunctions import find_element

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model("my_model.keras")

inputter = "2 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 298"

# Split the input string into individual number strings and convert to integers
lit = list(map(int, inputter.split()))

# Reshape the input to match the model's expected input shape (batch_size, 120)
input_array = np.array(lit).reshape(1, 120)

print(new_model.predict(input_array))