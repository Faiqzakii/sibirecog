import operator
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed

from utils import get_word_list


class SignDetector:
    def __init__(self, filepath=None):
        self.model = self.create_model() 
        if filepath is not None: self.model = self.load((filepath))
        

    def create_model(self):
        # Model tinggal diatur sesuai kebutuhan
        # Disini pake CNN biasa
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        
        # model = self.cnntd()

        # model.compile(optimizer='rmsprop',
        #                   loss='sparse_categorical_crossentropy',
        #                    metrics=['accuracy'])
        model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return model

    def cnntd(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 3, activation='relu'), input_shape=(None, 8, 2700)),
            tf.keras.layers.Conv2D(4, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4, activation='softmax'),
            tf.keras.layers.Dense(4, activation='relu')
        ])
        
        return model

    def load(self, filepath):
        model = tf.keras.models.load_model(filepath)
        return model

    def save(self, modelname='model'):
        self.model.save(f'model/{modelname}.h5')

    def train(self, x_input, y_input, epochs=100):
        # self.cp_callback = tf.keras.callbacks.EarlyStopping(monitor='loss')
        # self.model.fit(x_input, y_input, epochs=epochs, callbacks=[self.cp_callback])
        self.model.fit(x_input, y_input, epochs=epochs)
        print(self.model.summary())

    def evaluate(self, x_input):
        # probability_model = tf.keras.Sequential([
        #     self.model,
        #     tf.keras.layers.Softmax()
        # ])
        probability_model = self.model
        res = probability_model.predict(x_input)
        word_list = get_word_list()
        word_max_predict = {}
        for word in word_list:
            word_max_predict[word] = 0
        for predict in res:
            prob = predict[np.argmax(predict)]
            word_max_predict[word_list[np.argmax(predict)]] += 1
        # print(word_max_predict)
        # print(res)
        if prob <= 0.5:
            choosen_word = 'Tidak terdeteksi'
        else:
            choosen_word = max(word_max_predict.items(), key=operator.itemgetter(1))[0]
        return choosen_word, prob
