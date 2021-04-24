import os
import warnings

import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from image_utils import turnoff_irrelevant


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
        avg_ccolor = []

        for i, cat in enumerate(categories):
            catpath = '/'.join([dataset_path, cat])
            img_names = [name.name for name in os.scandir(
                catpath) if name.is_file()]
            class_imgs = []
            for j, file in enumerate(img_names):
                img = cv2.imread('/'.join([catpath, file]))
                img = cv2.resize(img, (self.input_dims[0], self.input_dims[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = turnoff_irrelevant(img)/255.
                class_imgs.append(img)

            train_imgs.extend(class_imgs)
            train_labels.extend(np.full(len(img_names), i), )
            #avg_ccolor.append(np.average(class_imgs, axis=(0, 1, 2)))

        x_train = np.array(train_imgs)
        y_train = np.array(train_labels)
        x_train, y_train = shuffle(x_train, y_train)

        return x_train, y_train

    def train(self, data_path='train_data/'):
        x_train, y_train = self.load_dataset(data_path)
        self.model.fit(x_train, y_train, epochs=25,
                       batch_size=len(x_train), verbose=1)
        #self.avg_ccolor = avg_ccolor

        #np.save('cached_models/avg_ccolor', avg_ccolor)

        self.model.save_weights(self.cache_path)

    def get_avg_ccolor(self):
        return self.avg_ccolor

    def color_distance(self, X):
        X_cavg = np.average(X, axis=(1, 2))
        return np.sum((X_cavg - self.avg_ccolor[1])**2, axis=1)

    def predict(self, X):
        x_arr = np.ndarray(shape=(len(X), *self.input_dims))
        for i in range(len(X)):
            x_arr[i] = turnoff_irrelevant(
                cv2.resize(X[i], self.input_dims[0:2]))/255.

        return self.model.predict(x_arr)

    def sharpen(self, img):
        return cv2.addWeighted(img, 4, cv2.blur(img, (30, 30)), -4, 128)

    def load_cached_model(self):
        #self.avg_ccolor = np.load('cached_models/avg_ccolor.npy')
        self.model.load_weights(self.cache_path)
