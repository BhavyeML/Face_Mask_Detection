#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def model_net():
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    
    return model

