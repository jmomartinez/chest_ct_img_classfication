import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class pre_processing():
    def __init__(self,train_path,valid_path,test_path,target_size,train_batch_size,valid_batch_size,test_batch_size):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.target_size = target_size
        self.train_size = train_batch_size
        self.valid_size = valid_batch_size
        self.test_size = test_batch_size

    def init_generators(self):
        target = self.target_size

        train_size = self.train_size
        valid_size = self.valid_size
        test_size = self.test_size

        train_it = ImageDataGenerator(vertical_flip=True)
        train_gen = train_it.flow_from_directory(self.train_path,batch_size=train_size,
        shuffle=True,target_size=target)

        valid_it = ImageDataGenerator()
        valid_gen = valid_it.flow_from_directory(self.valid_path,batch_size=valid_size,
        shuffle=True,target_size=target)

        test_it = ImageDataGenerator()
        test_gen = test_it.flow_from_directory(self.test_path,batch_size=test_size,
        shuffle=True,target_size=target)

        return train_gen,valid_gen,test_gen
