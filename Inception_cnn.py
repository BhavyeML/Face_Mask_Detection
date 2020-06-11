#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import RMSprop
import tensorflow_hub as hub



def inception_net():
    
    
    module_selection = ("inception_v3", 150) #@param ["(\"mobilenet_v2_100_224\", 224)", "(\"inception_v3\", 299)"] {type:"raw", allow-input: true}
    handle_base, pixels = module_selection
    MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)
    IMAGE_SIZE = (pixels, pixels)
    print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))
    BATCH_SIZE =  10#@param {type:"integer"}
    do_fine_tuning = True #@param {type:"boolean"}
    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        # Explicitly define the input shape so the model can be properly
        # loaded by the TFLiteConverter
        tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(50,activation='relu'),
        tf.keras.layers.Dense(2,kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))


    model.compile(
      optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
      metrics=['accuracy'])    
    return model

