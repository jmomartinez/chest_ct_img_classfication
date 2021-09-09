import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import image
import json


class eda_sda():
    def __init__(self,img_path,folder_path,batch_size):
        self.img_path = img_path
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.train_it = 0
        self.json_str = 0
        self.classes = ['adenocarcinoma','large cell carcinoma','normal','squamous cell carcinoma']

    def init_generator(self):
        train_gen = ImageDataGenerator()
        train_generator = train_gen.flow_from_directory(self.folder_path,batch_size=25)
        self.train_it = train_generator

    # Statistical Data Analysis
    def stats(self):
        # Single Image stats
        img = image.imread(self.img_path) # Read it in:matplotlib
        img_shape = img.shape

        img_info = {'Img Shape':img.shape,'Array Data Type':str(img.dtype),
        'Dimensions':[img_shape[0],img_shape[1]],'Channels':img_shape[2],
        'Max Val':str(img[0].max()),'Min Val':str(img[0].min())}

        # Pixel Stats
        img = Image.open(self.img_path)
        img = np.asarray(img.convert(mode='L'))

        # df = pd.DataFrame(img)
        # pix_stats = dict(df.describe())
        # img_info['Pixel Stats'] = pix_stats

        # Batch Stats
        features,target = next(self.train_it)

        batch_info = {'Features dtype':str(type(features)),'Features Shape':features.shape,
        'Target Shape':target.shape,'Class Indices':self.train_it.class_indices}
        
        # Batch Stats
        info = {**img_info,**batch_info}
        with open('data_stats.json','w') as write_file:
            json.dump(info,write_file,indent=4)
        json_str = json.dumps(info,indent=4)
        self.json_str = json_str

     # Exploratory Data Analysis
    def display_img(self):
        path = self.img_path
        data = image.imread(path)
        data = data[:,:,0]
        plt.imshow(data)
        plt.colorbar()
        plt.title('Chest CT({})'.format(self.classes[0]))
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        plt.show()

    def display_batch(self):
        features,target = next(self.train_it)

        plt.figure(figsize=(12,12))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(features[i].astype('uint8'))
            plt.suptitle('Chest CTs({})'.format(self.classes[0]),size=20)
        plt.show()