import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from model import multi_unet_model
import tensorflow as tf
from sklearn import preprocessing
import keras
from keras import backend as K

#prep training images and masks
train_folder='data/train/'
imnames=os.listdir(train_folder+'images')

#these will hold the images
train_images=[]
train_masks=[]

#create an image and mask list
for imname in imnames:
    img=cv2.imread(train_folder+'/images/'+imname,0)
    scaler=preprocessing.MinMaxScaler()
    img=scaler.fit_transform(img)
    train_images.append(img)

    mask=cv2.imread(train_folder+'/masks/'+imname.replace('jpg','png'),0)
    train_masks.append(mask)
    
#make sure it's a numpy array
train_images=np.array(train_images)
train_masks=np.array(train_masks)


#prep test images and masks
test_images=[]
test_masks=[]

test_folder='data/test/'
imnames=os.listdir(test_folder+'images')

for imname in imnames:
    img=cv2.imread(test_folder+'/images/'+imname,0)
    scaler=preprocessing.MinMaxScaler()
    img=scaler.fit_transform(img)
    test_images.append(img)
    
    mask=cv2.imread(test_folder+'/masks/'+imname,0)
    test_masks.append(mask)

test_images=np.array(test_images)
test_masks=np.array(test_masks)
    
  


#possibly not necessary. makes sure labels are 0,1,2....
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()
n,h,w=train_masks.shape
train_masks_reshaped=train_masks.reshape(-1,1)
train_masks_reshaped_encoded=label_encoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape=train_masks_reshaped_encoded.reshape(n,h,w)

#give the finla layer 6 dimensions

from tensorflow.keras.utils import to_categorical

train_masks_input=np.expand_dims(train_masks_encoded_original_shape,axis=3)
train_masks_cat=to_categorical(train_masks_input,num_classes=6)
y_train_cat=train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], 6))


#prepare x

train_images=np.expand_dims(train_images,axis=3)
x_train=train_images

#class weights
from sklearn.utils import class_weight

#calculate class weights
class_weights=class_weight.compute_class_weight('balanced',
                                                np.unique(train_masks_reshaped_encoded),
                                                train_masks_reshaped_encoded)

class_weights={i:class_weights[i] for i in range(5)}
print('Class weights are ...', class_weights)



#build model
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 128,128,1

model=multi_unet_model(6, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#define custom loss function
weights=[49,1,1,2,0.25,5]
def weighted_categorical_crossentropy(weights):
    def wcce(y_true, y_pred):
        Kweights = tf.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = tf.constant(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce    
loss = weighted_categorical_crossentropy(weights)
model.compile(optimizer='adam', loss=loss)
    
history=model.fit(x_train,
                  y_train_cat,  
                  batch_size=16,
                  verbose=1,
                  epochs=50,
#                  #validation_data=(test_images,test_masks),
#                  sample_weight=class_weights,
                  shuffle=False)