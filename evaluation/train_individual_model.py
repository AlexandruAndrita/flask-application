from sklearn.model_selection import train_test_split
import numpy as np


def find_model_output(regressor, known_array, image, train_size=0.02):
    height, width = image.shape
    flat_image = image.reshape(-1)
    x_coordinates = np.arange(len(flat_image)) % width
    y_coordinates = np.arange(len(flat_image)) // width
    data = np.array([x_coordinates, y_coordinates]).T
    target = flat_image
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=train_size, random_state=42)
    mean = y_train.mean()
    regressor.fit(X_train, y_train - mean)
    flat_picture = regressor.predict(data) + mean

    final_image = np.copy(image)

    flat_picture = flat_picture.reshape(image.shape[0],image.shape[1])

    final_image[known_array == 0] = flat_picture[known_array == 0]

    return final_image
