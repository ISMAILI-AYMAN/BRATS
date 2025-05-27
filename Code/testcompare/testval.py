import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from keras.models import load_model
import os
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)
my_model2=load_model(r'models\brats_nnUnet.hdf5',custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)}, 
                      compile=False)
img_num= np.linspace(0, 124,124, dtype=int) # 0-124
for i in img_num:
    test_img = np.load("BraTS2020_ValidationData/input_data_3channels/images/"+"image_"+str(i)+".npy")


    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction2 = my_model2.predict(test_img_input)
    test_prediction_argmax2=np.argmax(test_prediction2, axis=-1)[0,:,:,:]
    print(test_prediction_argmax2.shape)
    print(np.unique(test_prediction_argmax2))

    #Plot individual slices from test predictions for verification
    from matplotlib import pyplot as plt
    import random

    #_slice=random.randint(0, test_prediction_argmax.shape[2])
    #  Ensure n_slice is within bounds
    n_slice = 55
    '''
    # Plot the predictions
    plt.figure(figsize=(16, 10))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')
    plt.subplot(232)
    plt.title('Prediction on test2 image')
    plt.imshow(test_prediction_argmax2[:, :, n_slice])
    plt.show()'''
    #saveimg as npy
    make_dir = "BraTS2020_ValidationData/input_data_3channels/predictions/"
    if not os.path.exists(make_dir):
     os.makedirs(make_dir)
    # Save the prediction as a numpy file
    np.save("BraTS2020_ValidationData/input_data_3channels/predictions/prediction_"+str(i)+".npy", test_prediction_argmax2)
