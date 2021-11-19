"""
Created on 10/21/2021
Author: Zitong Yu
Contact: yu.zitong@wustl.edu

A simple Deep Convolutional Neural Network for denoising
Keras implementation
CPU-based

Python requirements
This code was tested on:
1. Python 3.6
2. TensorFlow 1.10.0
3. Keras 2.2.4
4. NumPy 1.19.2
"""

###############################################################################
#                            Import Libraries
###############################################################################
import numpy as np
import tensorflow as tf

from keras import backend as K

K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.regularizers import l2
from keras.models import load_model

###############################################################################
#                            Model
###############################################################################

#  Define loss function
def loss_fn(y_true, y_pred):
    # mean square error
    data_fidelity = tf.reshape(y_true, shape=[-1, 64*64]) - tf.reshape(y_pred, shape=[-1, 64*64])
    data_fidelity = tf.reduce_mean(tf.square(data_fidelity))
    return data_fidelity

# Repeating layers throughout the network 
def add_common_layers(filters, kernelsize,std, layer, bias_ct=0.03, leaky_alpha=0.01, drop_prob=0.1):
    if std == 2:
        pad = 'valid'
    else:
        pad = 'same'
    layer = Conv2D(filters, # num. of filters
                   kernel_size=kernelsize,
                   strides=std,
                   padding=pad,
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct),
                   kernel_regularizer=l2(0.1),
                   bias_regularizer=l2(0.1))(layer)
    layer = LeakyReLU(alpha=leaky_alpha)(layer) # activation func. 
    layer = Dropout(drop_prob)(layer) 
    return layer


def get_cnn():
    # This model has skip connections in place, here we use element-wise addition.
    # Define Convolutional Neural Network
    # Input shape 
    input = Input(shape=(64,64,1))
    conv1 = add_common_layers(16,(3, 3),1,layer=input)
    x = add_common_layers(16,(2, 2),2,conv1)
    conv2 = add_common_layers(32,(3, 3),1, x)
    x = add_common_layers(32,(2, 2),2,conv2)
    x = add_common_layers(64,(3, 3),1,x)
    # Transposed Convolution (upsampling)
    x = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Add()([x, conv2])
    x = add_common_layers(32,(3, 3), 1, x)
    # Transposed Convolution (upsampling)
    x = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Add()([x, conv1])
    x = add_common_layers(16,(3, 3),1, x)
    x = Conv2D(1, (3,3), strides=1, padding='same', use_bias=True, kernel_initializer='glorot_normal', bias_initializer=Constant(value=0.03),kernel_regularizer=l2(0.1),bias_regularizer=l2(0.1))(x)
    x = LeakyReLU(alpha=0.00)(x) # negative to zero
    output = x
    model = Model(inputs=[input], outputs=[output])
    return model

params = {'dim': (64,64), # image size
          'batch_size': 32,
          'n_channels': 1, # num. of channels
          'shuffle': True}

model = get_cnn()
model.summary()
model.compile(loss=loss_fn, optimizer='adam')

###############################################################################
#                            	    Training
###############################################################################
print('Loading the training data...')
num_train = 5000 # num. of training images. Change this number to match the actual number of images.
num_epochs = 500 # num. of epochs.

# import images here
X = np.zeros((num_train, *params['dim'], params['n_channels'])) # input data (low dose noisy images)
Y = np.zeros((num_train, *params['dim'], params['n_channels'])) # label (normal dose images)

# Images should be saved as 64X64 .npy files. Use provided writeNPY() MATLAB function to save images as .npy files.
for i in np.arange(num_train):
    X[i,:,:,0] = np.load(f'./data/train/low_dose/image{i}.npy') # You can change this path to your actual path.
    Y[i,:,:,0] = np.load(f'./data/train/normal_dose/image{i}.npy') # You can change this path to your actual path.

print('Training...')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# This callback will stop the training when there is no improvement in the validation loss for ten consecutive epochs. This callback function can prevent overfitting.
history = model.fit(x=X,
                    y=Y,
                    batch_size=params['batch_size'],
                    verbose=1,
                    validation_split=0.2,
                    epochs=num_epochs,
                    shuffle=params['shuffle'],
                    callbacks=[callback]) 
print('Training finished.')
train_loss = history.history['loss'] # training loss
val_loss = history.history['val_loss'] # validation loss

model.save(f'model_epochs{num_epochs}.h5') # save the trained model
np.save(f'train_loss_epochs{num_epochs}.npy',train_loss) # training loss saved in npy file. Use provided readNPY() MATLAB function to read.
np.save(f'val_loss_epochs{num_epochs}.npy',val_loss)# validation loss saved in npy file. Use provided readNPY() MATLAB function to read.

###############################################################################
#                            	    Prediction
###############################################################################
params = {'dim': (64,64),
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': True}

print('Loading the test data...')
num_train = 5000 # num. of test images. Change this number to match the actual number of images.
X_test = np.zeros((num_train, *params['dim'], params['n_channels']))
for i in np.arange(num_train):
    X_test[i,:,:,0] = np.load(f'./data/test/low_dose/image{i}.npy') # You can change this path to your actual path.

pred = model.predict(x=X_test,
                     batch_size=params['batch_size'],
                     verbose=1)

print('Saving the predicted data...')
for i in np.arange(num_train):
    current_pred = pred[i]
    np.save(f'./data/prediction/image{i}.npy',current_pred) # You can change this path to your actual path.

