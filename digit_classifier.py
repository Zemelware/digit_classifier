import os
import math

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage


def preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Normalize data
    train_images = train_images / 255
    test_images = test_images / 255

    return train_images, train_labels, test_images, test_labels


def create_model(learning_rate):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(140, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, x, y, batch_size, epochs):
    model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True)


def plot_digit(image):
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.show()


def shift(img, sx, sy):
    # Shift the image to be in the center
    rows, cols = img.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))

    return shifted


def get_best_shift(img):
    # Get the center of mass of the image
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def center_digit(img):
    # Remove every row and column at the sides of the image which are black
    # This re-formats the image to have no padding
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    rows, cols, _ = img.shape

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    # Pad the image array with 0s so that it's 28x28
    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rows_padding, cols_padding), "constant")
    plot_digit(img)

    # Center the image using its center of mass
    shiftx, shifty = get_best_shift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted
    plot_digit(img)

    return img


def preprocess_image(img):
    # Load image
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    # Convert to array
    img = keras.preprocessing.image.img_to_array(img)
    # Normalize the data
    img = img / 255
    # Center the image to make it looking like the training data.
    # This improves accuracy drastically
    img = center_digit(img)

    return img


def load_prediction_data(directory, f=None):
    images = []
    os.path.join(directory)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # If a file is specified, look for that file and return it. If no file is specified,
            # an array of all the images in the folder will be returned
            if (f is not None and filename == f) or f is None:
                img = os.path.join(directory, filename)
                img = preprocess_image(img)
                images.append(img)
    return np.array(images)


def predict_digit(folder, filename):
    predict_img = load_prediction_data(folder, filename)
    prediction = model.predict(predict_img)
    # print(str(np.argmax(prediction)), ["{:.2%}".format(i) for i in prediction[0]])
    return str(np.argmax(prediction))


train_images, train_labels, test_images, test_labels = preprocess_data()

# Hyperparameters
learning_rate = 0.01
epochs = 40
batch_size = 1000

# model = create_model(learning_rate)
# train_model(model, train_images, train_labels, batch_size, epochs)

# model.save("model.h5")
model = keras.models.load_model("model.h5")

# model.evaluate(test_images, test_labels, batch_size)

# predictions = model.predict(predict_images)

# for i in range(len(predict_images)):
#     plt.grid(False)
#     plt.imshow(predict_images[i], cmap=plt.cm.binary)
#     plt.title("Prediction: " + str(np.argmax(predictions[i])))
#     plt.show()
