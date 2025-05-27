# https://youtu.be/PNqnLbzdxwQ
"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators. 

No image processing operations are performed here, just load data from local directory
in batches. 

"""

#from tifffile import imsave, imread
import os
import numpy as np


def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir+image_name)
                      
            images.append(image)
    images = np.array(images)
    
    return(images)




def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

    # keras needs the generator infinite, so we will use while true  
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
                       
            X = load_img(img_dir, img_list[batch_start:limit]).astype(np.float32)  # Cast to float32
            Y = load_img(mask_dir, mask_list[batch_start:limit]).astype(np.float32)  # Cast to float32

            yield (X, Y)  # A tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

def imageLoader_with_rpn(img_dir, img_list, mask_dir, mask_list, batch_size):
    while True:
        for i in range(0, len(img_list), batch_size):
            batch_img = []
            batch_mask = []
            batch_rpn_class = []
            batch_rpn_bbox = []

            for j in range(i, i + batch_size):
                if j >= len(img_list):
                    break
                img = np.load(os.path.join(img_dir, img_list[j]))
                mask = np.load(os.path.join(mask_dir, mask_list[j]))

                # Generate dummy RPN class and bbox data (replace with actual logic)
                rpn_class = np.zeros((16, 16, 16, 4))  # Example shape
                rpn_bbox = np.zeros((16, 16, 16, 6))  # Example shape

                batch_img.append(img)
                batch_mask.append(mask)
                batch_rpn_class.append(rpn_class)
                batch_rpn_bbox.append(rpn_bbox)

            yield (
                np.array(batch_img),
                {
                    'rpn_class': np.array(batch_rpn_class),
                    'rpn_bbox': np.array(batch_rpn_bbox),
                    'mask_output': np.array(batch_mask),
                },
            )
############################################

#Test the generator

from matplotlib import pyplot as plt
import random

train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

batch_size = 2

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()


img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()
