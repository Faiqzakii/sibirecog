import operator
import os

import numpy as np
import tensorflow as tf

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

        # Contoh RNN
        # self.model = tf.keras.models.Sequential([
        #     tf.keras.layers.SimpleRNN(64),
        #     tf.keras.layers.Dense(64, activation='softmax'),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.Dense(128, activation='linear'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(21, activation='relu')
        # ])
        model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return model

    def load(self, filepath):
        model = tf.keras.models.load_model(filepath)
        return model

    def save(self, modelname='model'):
        self.model.save(f'model/{modelname}.h5')

    def train(self, x_input, y_input, epochs=100):
        #self.model.fit(x_input, y_input, epochs=epochs, callbacks=[self.cp_callback])
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
        choosen_word = max(word_max_predict.items(), key=operator.itemgetter(1))[0]
        return choosen_word, prob
