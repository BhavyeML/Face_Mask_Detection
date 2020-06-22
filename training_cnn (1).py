#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import glob
import re
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
print(tf.__version__)

#Listing all the files present
dirr=os.getcwd()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#Checking for availability of GPU
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True


# In[ ]:


class augment(object):
    
    def __init__(self, DATA_DIR, rescale,rotation_range,width_shift_range,height_shift_range,shear_range,zoom_range,horizontal_flip,fill_mode,batch_size,class_mode):
        """
        
            Input: DATA_DIR: Data Directory 
                   Others: Various variable to be used for In-Memory Augmentation
                   
        """
        
        self.rescale=rescale
        if self.rescale is None:
            self.rescale=1.0/255.
            
        self.rotation_range=rotation_range
        if self.rotation_range is None:
            self.rotation_range=40
            
        self.width_shift_range=width_shift_range
        if self.width_shift_range is None:
            self.width_shift_range=0.2
            
        self.height_shift_range=height_shift_range
        if self.height_shift_range is None:
            self.height_shift_range=0.2
            
        self.shear_range= shear_range
        if self.shear_range is None:
            self.shear_range=0.2
        
        self.zoom_range= zoom_range
        if self.zoom_range is None:
            self.zoom_range=0.2
            
        self.horizontal_flip=horizontal_flip
        if self.horizontal_flip is None:
            self.horizontal_flip=True
            
        self.fill_mode=fill_mode
        if self.fill_mode is None:
            self.fill_mode='nearest'
            
        self.batch_size=batch_size
        if self.batch_size is None:
            self.batch_size=20
            
        self.class_mode= class_mode
        if self.class_mode is None:
            self.class_mode='binary'
        
        
        self.train_dir = os.path.join(DATA_DIR,"train")
        if not os.path.isdir(self.train_dir):
            raise IOError(f"{self.train_dir} doesn't exist")
            
        self.val_dir = os.path.join(DATA_DIR,"validation")
        if not os.path.isdir(self.val_dir):
            raise IOError(f"{self.val_dir} doesn't exist")
    
    def train_aug(self):
        
        """ Image Augementation in Training Images"""
        train_datagen=ImageDataGenerator(rescale=self.rescale, rotation_range=self.rotation_range,
                                width_shift_range=self.width_shift_range,
                                height_shift_range=self.height_shift_range,
                                shear_range=self.shear_range,
                                zoom_range=self.zoom_range,
                                horizontal_flip=self.horizontal_flip,
                                fill_mode=self.fill_mode)
        
        train_generator=train_datagen.flow_from_directory(self.train_dir,target_size=(300,300),batch_size=self.batch_size,
                                                 class_mode=self.class_mode)
        
        return train_generator
    
    def val_aug(self):
        
        """ Image Augementation in Validation Images"""
        
        val_datagen=ImageDataGenerator(rescale=self.rescale)
        
        val_generator=val_datagen.flow_from_directory(self.val_dir,target_size=(300,300),batch_size=10,class_mode=self.class_mode)
        return val_generator


# In[ ]:


class Net(object):
 
    def __init__(self, input_dim,dropout):
        """  
        Input: 
        
                input_dim:size of image
                batch_size
        """
        
        self.input_dim=input_dim
        if self.input_dim is None:
            self.input_dim=(300,300,3)
        
        self.dropout=dropout
        if self.dropout is None:
            self.dropout=0.2
            
        #self.batch_size=batch_size
        #if self.batch_size is None:
            #self.batch_size=64
      
        print(" Model Training Initiated")
        print(" image_size:", self.input_dim)
        print(" dropout:", self.dropout)
        
    
    def model_cnn(self):
        
        """ 
            a). Model Trained From Scratch With 3 CNN and 3 Max Pool Layers 
            b). Model Compliation with RMSProp Algorithm and binary_crossentropy loss function
            
        """
        
        from tensorflow.keras.optimizers import RMSprop

        model=tf.keras.models.Sequential([
                                        tf.keras.layers.InputLayer(input_shape=self.input_dim),
                                        tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
                                        tf.keras.layers.BatchNormalization(axis=-1),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Dropout(self.dropout),
                                        tf.keras.layers.Conv2D(256,(2,2),activation='relu'),
                                        tf.keras.layers.BatchNormalization(axis=-1),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Dropout(self.dropout),
                                        tf.keras.layers.Conv2D(1024,(3,3),activation='relu'),
                                        tf.keras.layers.BatchNormalization(axis=-1),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dropout(self.dropout),
                                        tf.keras.layers.Dense(512,activation='relu'),
                                        tf.keras.layers.BatchNormalization(axis=-1),
                                        tf.keras.layers.Dense(128,activation='relu'),
                                        tf.keras.layers.Dense(1,activation='sigmoid')
                ])
        model.summary()


        model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy', metrics=['acc'])
        
        return model


# In[ ]:


class myCallback(tf.keras.callbacks.Callback):
    
    """
        Function: Used to Stop Training When Desired Accuracy is Achieved
        
    """
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


# In[ ]:


class trainer(augment,Net,object):
    
    def __init__(self,epoch, input_dim,dropout,batch_size,class_mode,DATA_DIR, rescale=1.0/255.,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest'):
        
        Net.__init__(self,input_dim,dropout)
        augment.__init__(self,DATA_DIR, rescale,rotation_range,width_shift_range,height_shift_range,shear_range,zoom_range,horizontal_flip,fill_mode,batch_size,class_mode)
        
        self.batch_size=batch_size
        if self.batch_size is None:
            self.batch_size=64
            
        self.class_mode=class_mode
        if self.class_mode is None:
            self.class_mode='binary'
        
        self.epochs=epoch
        if self.epochs is None:
            self.epochs=100
            
    
    def model_cnn(self):
        
        """
            Function: Inherits Model from Class Net
            
        """
        model=Net.model_cnn(self)
        return model
    
   
    def train_aug(self):
        
        """
            Function: Inherits Augmented Function for Training Images from Class Augment
            
        """
        train_gen= augment.train_aug(self)
        return train_gen
        
    def val_aug(self):
        
        """
            Function: Inherits Augmented Function for Validation Images from Class Augment
            
        """
        val_gen=augment.val_aug(self)
        return val_gen
    
    def train(self):
        
        """
            Function: Trains and saves the model
        
        """
        
        model=self.model_cnn()
        train_generator=self.train_aug()
        test_generator=self.val_aug()
        
        callbacks=myCallback()
        device_name = tf.test.gpu_device_name()
        if "GPU" not in device_name:
            print("GPU device not found")
            history = model.fit_generator(train_generator,epochs=self.epochs,validation_data=test_generator,verbose=2,callbacks=[callbacks])
        else:
            print('Found GPU at: {}'.format(device_name))
            config = tf.compat.v1.ConfigProto() 
            config.gpu_options.allow_growth = True
            with tf.device('/gpu:0'):
                history = model.fit_generator(train_generator,epochs=self.epochs,validation_data=test_generator,validation_steps = 26,verbose=2,callbacks=[callbacks])
         
            
            
        
        model.save('final_model_train1.hdf5')
        self.visualize(history)
    
    def visualize(self,history):
        """
            Function: Outputs Plots of Model's Metric
        
        """
        acc=history.history['acc']
        val_acc=history.history['val_acc']
        epochs=range(len(acc))
        plt.plot(epochs,acc)
        plt.plot(epochs,val_acc)
        plt.show()


# In[ ]:


if __name__=="__main__":
    path_dir=os.path.join(os.getcwd(),"horse-or-human")
    tester=trainer(epoch=100,input_dim=(300,300,3),dropout=0.2,batch_size=20,class_mode='binary',DATA_DIR=path_dir,rescale=1.0/255.,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    tester.train()


# In[ ]:




