from sklearn.model_selection import train_test_split

def split(X, y, test_size):
    X_training, X_validation, y_training, y_validation = train_test_split(X, y, test_size=test_size, random_state=True)
