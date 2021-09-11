import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class pre_processing():
    def __init__(self,train_path,valid_path,test_path,batch_size,target_size):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.target_size = target_size

    def init_generators(self):
        tsize = self.target_size
        bsize = self.batch_size

        train_it = ImageDataGenerator(vertical_flip=True)
        train_gen = train_it.flow_from_directory(self.train_path,batch_size=bsize,
        shuffle=True,target_size=tsize)

        valid_it = ImageDataGenerator()
        valid_gen = valid_it.flow_from_directory(self.valid_path,batch_size=bsize,
        shuffle=True,target_size=tsize)

        test_it = ImageDataGenerator()
        test_gen = test_it.flow_from_directory(self.test_path,batch_size=sbsize,
        shuffle=True,target_size=tsize)

        return train_gen,valid_gen,test_gen
