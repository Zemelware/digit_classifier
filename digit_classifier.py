import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


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


def load_prediction_data(directory, f=None):
    images = []
    os.path.join(directory)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # If a file is specified, look for that file and return it. If no file is specified,
            # an array of all the images in the folder will be returned
            if (f is not None and filename == f) or f is None:
                img = os.path.join(directory, filename)
                # Load image
                img = keras.preprocessing.image.load_img(img, color_mode="grayscale", target_size=(28, 28))
                # Convert to array
                img = keras.preprocessing.image.img_to_array(img)
                # Reshape the array into the proper size
                img = np.array(img).reshape(28, 28)
                # Normalize the data
                img = img / 255
                images.append(img)
    return np.array(images)


def predict_digit(folder, filename):
    predict_img = load_prediction_data(folder, filename)
    prediction = model.predict(predict_img)
    # print(str(np.argmax(prediction)), ["{:.2%}".format(i) for i in prediction[0]])
    return str(np.argmax(prediction))


train_images, train_labels, test_images, test_labels = preprocess_data()
# predict_images = load_prediction_data(dir="numbers")

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
