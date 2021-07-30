import operator
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow import keras

from utils import get_word_list

class SignDetector:
    def __init__(self, filepath=None):
        if filepath is not None: self.model = self.load(filepath)
        

    def create_model(self, x_train, ny, arch = 'seqlstm'):

        archkey = {
            'bilstm'    : self.bilstm(x_train, ny),
            'seqlstm'   : self.seqlstm(x_train, ny),
            'seqlstm2'  : self.seqlstm2(x_train, ny),
            'seqlstm3'  : self.seqlstm3(x_train, ny),
            'cnn'       : self.cnn(x_train, ny),
            'connet'    : self.connet(x_train, ny),
            'liulstm'   : self.liulstm(x_train, ny),
            'zhanglstm' : self.zhanglstm(x_train, ny),
            'zhanggru'  : self.zhanggru(x_train, ny),
            'zhanggru2'  : self.zhanggru2(x_train, ny),
            'zhanggru3'  : self.zhanggru3(x_train, ny)
        }
        
        model = archkey[arch]
        
        model.compile(loss='binary_crossentropy', 
                      optimizer='Adamax',
                      metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
        
        model.summary()
        self.model = model
    
    def dnn(self, ny):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(ny, activation='softmax')
        ])
        
        return model
        
    def cnn(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        cnntd = Sequential(Conv1D(filters=64, kernel_size=3, activation='relu'))(inputs)
        cnntd = Dropout(0.2)(cnntd)
        cnntd = Sequential(MaxPool1D(pool_size=2))(cnntd)
        cnntd = Dropout(0.2)(cnntd)
        cnntd = Flatten()(cnntd)
        cnntd = Dense(128, activation='relu')(cnntd)
        cnntd = Dropout(0.4)(cnntd)
        class_output = Dense(ny, activation='softmax', name='class_output')(cnntd)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def connet(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        vnet = Sequential(Conv1D(filters=256, kernel_size=2, activation='relu'))(inputs)
        vnet = Sequential(MaxPool1D(pool_size=2))(vnet)
        vnet = Dropout(0.2)(vnet)
        vnet = Sequential(Conv1D(filters=256, kernel_size=2, activation='relu'))(vnet)
        vnet = Sequential(MaxPool1D(pool_size=2))(vnet)
        #vnet = Dropout(0.2)(vnet)
        #vnet = Sequential(Conv1D(filters=16, kernel_size=2, activation='relu'))(vnet)
        #vnet = Sequential(MaxPool1D(pool_size=2))(vnet)
        #vnet = Sequential(Conv1D(filters=16, kernel_size=2, activation='relu'))(vnet)
        #vnet = Sequential(MaxPool1D(pool_size=2))(vnet)
        vnet = Dropout(0.2)(vnet)
        vnet = Flatten()(vnet)
        class_output = Dense(ny, activation='softmax', name='class_output')(vnet)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def liulstm(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        seq = Sequential(LSTM(512))(inputs)
        seq = Dense(512)(seq)
        seq = Dense(100)(seq)
        class_output = Dense(ny, activation='softmax')(seq)
        
        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def zhanglstm(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        seq = Sequential(LSTM(500, return_sequences = True))(inputs)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax')(seq)
        
        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def zhanggru(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        seq = Sequential(GRU(500, return_sequences = True))(inputs)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax')(seq)
        
        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def zhanggru2(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        seq = Sequential(GRU(100, return_sequences = True))(inputs)
        seq = Sequential(GRU(500, return_sequences = True))(seq)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax')(seq)
        
        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def zhanggru3(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
        
        seq = Sequential(GRU(100, return_sequences = True))(inputs)
        seq = Sequential(GRU(300, return_sequences = True))(seq)
        seq = Sequential(GRU(500, return_sequences = True))(seq)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax')(seq)
        
        model = keras.models.Model(inputs=inputs, outputs=class_output)
        
        return model
    
    def seqlstm(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        seq = Sequential(LSTM(128, return_sequences=True))(inputs)
        seq = Dropout(0.2)(seq)
        seq = Sequential(LSTM(256, return_sequences=True))(seq)
        seq = Dropout(0.2)(seq)
        seq = Sequential(LSTM(256, return_sequences=True))(seq)
        seq = Dropout(0.2)(seq)
        seq = Sequential(LSTM(128, return_sequences=True))(seq)
        seq = Dropout(0.2)(seq)
        seq = Sequential(LSTM(256, return_sequences = True))(seq)
        seq = Dropout(0.2)(seq)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax', name='class_output')(seq)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        return model
    
    def seqlstm2(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        seq = Sequential(LSTM(128, return_sequences=True))(inputs)
        seq = Sequential(LSTM(64, return_sequences=True))(seq)
        seq = AveragePooling1D()(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax', name='class_output')(seq)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        return model

    def seqlstm3(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        seq = Sequential(LSTM(128, return_sequences=True))(inputs)
        seq = Sequential(LSTM(64, return_sequences=True))(seq)
        seq = AveragePooling1D()(seq)
        seq = Dropout(0.2)(seq)
        seq = Flatten()(seq)
        class_output = Dense(ny, activation='softmax', name='class_output')(seq)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        return model
    
    def bilstm(self, x_train, ny):
        inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))

        bidirect = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        bidirect = Dropout(0.2)(bidirect)
        bidirect = Bidirectional(LSTM(256, return_sequences=True))(bidirect)
        bidirect = Dropout(0.2)(bidirect)
        bidirect = Bidirectional(LSTM(256, return_sequences=True))(bidirect)
        bidirect = Dropout(0.2)(bidirect)
        bidirect = Bidirectional(LSTM(128, return_sequences=True))(bidirect)
        bidirect = Dropout(0.2)(bidirect)
        bidirect = Bidirectional(LSTM(256))(bidirect)
        bidirect = Dropout(0.2)(bidirect)
        class_output = Dense(ny, activation='softmax', name='class_output')(bidirect)

        model = keras.models.Model(inputs=inputs, outputs=class_output)
        return model

    def load(self, filepath):
        print(f'Loading pre-trained models from {filepath}')
        model = tf.keras.models.load_model(filepath)
        return model

    def save(self, modelname='model'):
        self.model.save(f'model/{modelname}.h5')

    def train(self, x_train, y_train, x_val, y_val, epochs=1000):
        self.es_callback = tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss', min_delta=0, patience=50, verbose=1,
                                mode='min', baseline=None, restore_best_weights=True
                            )
        
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 validation_data = (x_val, y_val),
                                 callbacks = [self.es_callback])
        return history
    
    def train_epoch(self, x_train, y_train, x_val, y_val, epochs=100):
        history = self.model.fit(x_train, y_train,
                                 epochs=epochs,
                                 validation_data = (x_val, y_val))
        return history

    def evaluate(self, x_test):
        probability_model = self.model
        words = get_word_list()
        token_labels = {i:words[i] for i in range(0, len(words))}
        
        probability_model = self.model
        pred = probability_model.predict(x_test)
        
        maxconf = np.argmax(pred[0])
        prob = pred[0][maxconf]
        
        choosen_word = token_labels[maxconf]
        
        return choosen_word, prob

    def multieval(self, x_test, y_test):
        words = get_word_list()
        token_labels = {i:words[i] for i in range(0, len(words))}
        
        probability_model = self.model
        pred = probability_model.predict(x_test)
        
        y_pred = [token_labels[np.argmax(p)] for p in pred]
        y_actl = [token_labels[np.argmax(p)] for p in y_test]
        conf   = [p[np.argmax(p)] for p in pred]
        
        d = pd.DataFrame({'Actual': y_actl, 'Predicted': y_pred, 'Confidence': conf})
        #d = d[d.columns[::-1]]
        print(d.to_string())
        
        cnt = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_actl[i]:
                cnt += 1 

        acc = cnt/len(y_actl)*100
        print(f'Accuracy: {acc}')
        return acc
        
        