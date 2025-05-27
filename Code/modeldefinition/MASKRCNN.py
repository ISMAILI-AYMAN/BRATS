import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation, Conv3DTranspose

def mask_rcnn_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))

    # Backbone (Feature Extraction)
    c1 = Conv3D(32, (3, 3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(64, (3, 3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(128, (3, 3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    # Region Proposal Network (RPN)
    rpn_conv = Conv3D(256, (3, 3, 3), padding='same')(p3)
    rpn_conv = BatchNormalization()(rpn_conv)
    rpn_conv = Activation('relu')(rpn_conv)

    # Classification Head
    rpn_class = Conv3D(num_classes, (1, 1, 1), activation='softmax', name='rpn_class')(rpn_conv)

    # Bounding Box Regression Head
    rpn_bbox = Conv3D(6, (1, 1, 1), activation='linear', name='rpn_bbox')(rpn_conv)

    # Mask Head
    mask_conv1 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(rpn_conv)
    mask_conv1 = BatchNormalization()(mask_conv1)
    mask_conv1 = Activation('relu')(mask_conv1)

    mask_conv2 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(mask_conv1)
    mask_conv2 = BatchNormalization()(mask_conv2)
    mask_conv2 = Activation('relu')(mask_conv2)

    # Upsample to match the ground truth resolution
    mask_output = Conv3DTranspose(num_classes, (2, 2, 2), strides=(2, 2, 2), padding='same', activation='softmax', name='mask_output')(mask_conv2)

    # Combine all outputs
    model = Model(inputs=inputs, outputs=[rpn_class, rpn_bbox, mask_output])
    return model