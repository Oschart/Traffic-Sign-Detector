import os
import warnings

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # To disable using GPU
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

CACHE_DIR = 'cached_models'
MODEL_NAME = 'sign_classifier'

class SignClassifier:
    def __init__(self, use_cached=False):
        self.cache_path = f'{CACHE_DIR}/{MODEL_NAME}'
        self.input_dims = (100, 100, 3)
        self.model = self.build_model()

        if use_cached:
            self.load_cached_model()
        else:
            self.train()

    def build_model(self):
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu,
                  input_shape=self.input_dims))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Conv2D(128, (3, 3), activation=tf.nn.relu))
        model.add(layers.MaxPool2D(2, 2))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation=tf.nn.relu))

        model.add(layers.Dense(1, activation=tf.nn.sigmoid))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    def load_dataset(self, dataset_path, as_gray=False):
        categories = [f.name for f in os.scandir(dataset_path) if f.is_dir()]

        train_imgs = []
        train_labels = []

        for i, cat in enumerate(categories):
            catpath = '/'.join([dataset_path, cat])
            img_names = [name.name for name in os.scandir(
                catpath) if name.is_file()]
            for j, file in enumerate(img_names):
                img = cv2.imread('/'.join([catpath, file]))
                img = cv2.resize(img, (self.input_dims[0], self.input_dims[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                train_imgs.append(img/255.)

            train_labels.extend(np.full(len(img_names), i), )

        x_train = np.array(train_imgs)
        y_train = np.array(train_labels)
        x_train, y_train = shuffle(x_train, y_train)

        return x_train, y_train

    def train(self, data_path='train_data/'):
        x_train, y_train = self.load_dataset(data_path)
        self.model.fit(x_train, y_train, epochs=25, verbose=1)

        self.model.save_weights(self.cache_path)

    def predict(self, X):
        x_arr = np.ndarray(shape=(len(X), *self.input_dims))
        for i in range(len(X)):
            x_arr[i] = cv2.resize(X[i], self.input_dims[0:2])/255.

        return self.model.predict(x_arr)


    def load_cached_model(self):
        self.model.load_weights(self.cache_path)
