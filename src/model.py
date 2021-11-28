from pre_process import pre_processing
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

class model():
    def __init__(self,epochs,in_shape,pool_size,kernel_size,train_gen,valid_gen,test_gen):
        self.epochs = epochs
        self.shape = in_shape
        self.p_size = pool_size
        self.k_size = kernel_size
        self.tgen = train_gen
        self.vgen = valid_gen
        self.test_gen = test_gen

    def create_model(self):
        inp = self.shape
        k = self.k_size
        p = self.p_size
        cnn = Sequential()

        # 3 Conv2D + MaxPool2D layer pairs
        cnn.add(Conv2D(filters=32,kernel_size=k,input_shape=inp,activation='relu'))
        cnn.add(MaxPool2D(pool_size=p))
        cnn.add(Conv2D(filters=64,kernel_size=k,activation='relu'))
        cnn.add(MaxPool2D(pool_size=p))
        cnn.add(Conv2D(filters=128,kernel_size=k,activation='relu'))
        cnn.add(MaxPool2D(pool_size=p))

        cnn.add(Flatten())
        cnn.add(Dense(128,activation='relu'))
        cnn.add(Dropout(rate=.15)) # Only applied during training (e.g. model.fit())
        cnn.add(Dense(64,activation='relu'))
        cnn.add(Dense(4,activation='softmax'))
        plot_model(cnn, to_file='train_model.png', show_shapes=True)

        cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return cnn

    def train_model(self):
        chk_point = ModelCheckpoint('best_weights.hdf5',monitor='val_loss',
                mode='min',verbose=1,save_best_only=True,save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=3,verbose=1)
        callbacks=[chk_point,early_stopping]

        cnn = self.create_model()
        print(self.tgen.n)
        trained_model = cnn.fit(self.tgen,epochs=self.epochs,steps_per_epoch=(self.tgen.n//self.tgen.batch_size),
                validation_data=self.vgen,validation_steps=(self.vgen.n//self.vgen.batch_size),callbacks=callbacks)
        return trained_model

    def final_model(self):
        cnn = self.create_model()
        cnn.load_weights('best_weights.hdf5')
        _,accuracy = cnn.evaluate(self.test_gen,steps=(self.test_gen.n//self.test_gen.batch_size))
        print('Evaluation Accuracy:',accuracy)

    def results(self):
        trained_model = self.train_model()

        plt.plot(trained_model.history['loss'])
        plt.plot(trained_model.history['val_loss'])
        plt.legend(['Train','Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epochs vs Loss')
        plt.show()