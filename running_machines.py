# Put in different functions to run the different machine learning models

# Gets the model that the user wants to load
# model_to_load = input("which model would you like to load:")


# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
import pandas as pd 
from sklearn.model_selection import train_test_split
 
# load model
model = load_model('my_model.keras')
# summarize model.
model.summary()

# load dataset
dataframe = pd.read_csv("information.csv", delimiter=",", header=None)
dataset = dataframe.values

# This tells you how many entries there are in matrix format (15541, 121)
# print(dataset.shape)

x_data = dataset[:, :-1]
y_data = dataset[:, -1]

X_training, X_validation, y_training, y_validation = train_test_split(x_data, y_data, test_size=0.75)

score = model.evaluate(X_validation, y_validation, verbose=2)
print(score)


