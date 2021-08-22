import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

imlength=128

def chop(imname,input_image_folder,input_mask_folder,output_image_folder,output_mask_folder):
    img=cv2.imread(input_image_folder+imname)
    mask=cv2.imread(input_mask_folder+imname)
    
    train_masks_testing=[]
    for i in range(0,6):
        for j in range(0,6):
            small_img=img[imlength*i:imlength*(i+1),imlength*j:imlength*(j+1)]
            cv2.imwrite(output_image_folder+str(i)+'-'+str(j)+'-'+imname,small_img)

            small_mask=mask[imlength*i:imlength*(i+1),imlength*j:imlength*(j+1)]
            cv2.imwrite(output_mask_folder+str(i)+'-'+str(j)+'-'+imname.replace('jpg','png'),small_mask)
    return train_masks_testing

input_image_folder='data/raw_images/'
input_mask_folder='data/raw_masks/'
output_image_folder='data/train/images/'
output_mask_folder='data/train/masks/'

for imname in ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg']:
    chop(imname,input_image_folder,input_mask_folder,output_image_folder,output_mask_folder)
    

output_image_folder='data/test/images/'
output_mask_folder='data/test/masks/'

#prep training images and masks
train_folder='data/train/'
imnames=os.listdir(train_folder+'images')

train_images=[]
train_masks=[]
