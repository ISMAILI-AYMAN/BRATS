import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from keras.models import load_model
import os
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"
train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
##################################
from custom_datagen import imageLoader  # Assuming you have a custom imageLoader function
import os  # If file paths are dynamically handled
my_model = load_model('brats_3d.hdf5',
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)}, 
                      compile=False)

my_model1=load_model('brats_3dRCNN.hdf5',custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)}, 
                      compile=False)
my_model2=load_model('brats_nnUnet.hdf5',custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)}, 
                      compile=False)

#Verify IoU on a batch of images from the test dataset
#Using built in keras function for IoU
#Only works on TF > 2.0
from keras.metrics import MeanIoU

batch_size=8 #Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
'''test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())'''

#############################################
#Predict on a few test images, one at a time
#Try images: 
img_num = 82

test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num)+".npy")

test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
print(test_prediction.shape)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]
# Predict using my_model1
test_prediction1 = my_model1.predict(test_img_input)
print("Number of outputs:", len(test_prediction1))
print(test_prediction1[0].shape)  # Check the shape of the first output
print(test_prediction1[1].shape)  # Check the shape of the second output
print(test_prediction1[2].shape)  # Check the shape of the third output

# Apply np.argmax along the last axis
test_prediction_argmax1 = np.argmax(test_prediction1[2], axis=-1)[0, :, :, :]


test_prediction2 = my_model2.predict(test_img_input)
test_prediction_argmax2=np.argmax(test_prediction2, axis=-1)[0,:,:,:]
print(test_prediction_argmax.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax))
print(test_prediction_argmax1.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax1))
print(test_prediction_argmax2.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax2))

#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#_slice=random.randint(0, test_prediction_argmax.shape[2])
# Ensure n_slice is within bounds
n_slice = min(55, test_prediction_argmax1.shape[2])

# Plot the predictions
plt.figure(figsize=(16, 10))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.subplot(234)
plt.title('Prediction on test1 image')
plt.imshow(test_prediction_argmax1[:, :, n_slice])  # Ensure n_slice is valid
plt.subplot(235)
plt.title('Prediction on test2 image')
plt.imshow(test_prediction_argmax2[:, :, n_slice])
plt.show()
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

# Flatten arrays for comparison
def flatten_and_compare(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return y_true_flat, y_pred_flat

# Compute IoU and Dice scores
def compute_scores(y_true, y_pred):
    y_true_flat, y_pred_flat = flatten_and_compare(y_true, y_pred)
    iou = jaccard_score(y_true_flat, y_pred_flat, average='weighted')
    dice = f1_score(y_true_flat, y_pred_flat, average='weighted')
    return iou, dice

# Compute scores for each model
iou_model, dice_model = compute_scores(test_mask_argmax, test_prediction_argmax)
iou_model1, dice_model1 = compute_scores(test_mask_argmax, test_prediction_argmax1)
iou_model2, dice_model2 = compute_scores(test_mask_argmax, test_prediction_argmax2)

# Print the scores
print("Scores for my_model:")
print(f"IoU: {iou_model:.4f}, Dice: {dice_model:.4f}")

print("\nScores for my_model1:")
print(f"IoU: {iou_model1:.4f}, Dice: {dice_model1:.4f}")

print("\nScores for my_model2:")
print(f"IoU: {iou_model2:.4f}, Dice: {dice_model2:.4f}")
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

# Appliquer Watershed sur une prédiction
def apply_watershed(prediction_argmax):
    # Calculer la distance transformée
    distance = ndimage.distance_transform_edt(prediction_argmax)
    
    # Trouver les maxima locaux (ajuster la dimension pour correspondre à la prédiction)
    local_maxi = peak_local_max(distance, labels=prediction_argmax.astype(np.int32), footprint=np.ones((3, 3, 3)), exclude_border=False)
    
    # Convertir les maxima locaux en un masque binaire avec la même forme que prediction_argmax
    local_maxi_binary = np.zeros_like(prediction_argmax, dtype=bool)
    local_maxi_binary[tuple(local_maxi.T)] = True  # Convert coordinates to binary mask
    
    # Marquer les maxima locaux
    markers = ndimage.label(local_maxi_binary)[0]
    
    # Vérifier que les dimensions des marqueurs et du masque correspondent
    if markers.shape != prediction_argmax.shape:
        raise ValueError(f"Shape mismatch: markers shape {markers.shape} and prediction_argmax shape {prediction_argmax.shape}")
    
    # Appliquer Watershed
    labels = watershed(-distance, markers, mask=prediction_argmax)
    return labels
# Exemple d'application sur une prédiction
test_prediction_watershed = apply_watershed(test_prediction_argmax)

# Visualiser les résultats
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Prediction (Before Watershed)')
plt.imshow(test_prediction_argmax[:, :, n_slice])
plt.subplot(232)
plt.title('Prediction (After Watershed)')
plt.imshow(test_prediction_watershed[:, :, n_slice])
plt.show()