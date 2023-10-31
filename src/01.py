import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
print(tf.__version__)

x= np.random.uniform(low=-10,high=10,size=(100,1))
y = np.random.uniform(low=-5,high=-5,size=(100,1))

noise = np.random.uniform(low=-1,high=1,size=(100,1))

input = np.column_stack((x,y))


model = keras.Sequential([
    keras.layers.Dense(units=1)])
model.compile(optimizer='sgd',
              loss= 'mean_squared_error',
              metrics= ['mse'])

model.fit(input,y,epochs=15,verbose=1,validation_split=0.2)

