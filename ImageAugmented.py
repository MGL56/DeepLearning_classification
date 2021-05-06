#!/usr/bin/env python
# coding: utf-8

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import glob


def imgAugmentor(images):

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-45, 45),
                shear=(-5, 5),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200)
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)),
                        iaa.AverageBlur(k=(3, 5)),
                        iaa.MedianBlur(k=(3, 5)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.8, 1.2)),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0.5, 1.0), direction=(0.5, 1.0)
                        ),
                    ])),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.01*255), per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.01, 0.05), size_percent=(0.01, 0.03),
                            per_channel=0.2
                        ),
                    ]),

                    # Invert each image's channel with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-5, 5), per_channel=0.5),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.8, 1.2), per_channel=0.5),

                    # Improve or worsen the contrast of images.
                    iaa.LinearContrast((0.8, 1.2), per_channel=0.5),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.0), sigma=0.25)
                    ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    images_aug = seq(images=images)
    return images_aug

class ImgDataGenerator(keras.utils.Sequence):
    
    def __init__(self, img_paths, labels, batch_size=64, target_size=(224, 224), augmentor=None, shuffle=False):
        'Generates data for Keras'
        self.X            = img_paths           # array of labels
        self.y            = labels              # array of image paths
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.on_epoch_end()
        self.dim          = target_size
        self.augmentor    = augmentor
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        y = np.array([self.y[k] for k in indexes])
        X = [np.array(image.load_img(self.X[k], target_size=self.dim)) for k in indexes]
        
        if self.augmentor != None:
            X = self.augmentor(X)
        
        X = np.array([preprocess_input(x) for x in X])
        return X, y

class CreateImgDataGenerator:
    
    def __init__(self, batch_size=48, seed=None):
        
        self.seed = seed
        self.batch_size = batch_size
        
        
    def create_train_test_csv(self, directory, n_classes, prefix='', encoder=LabelEncoder(), test_split=0.2):
        
        folders = os.listdir(directory)
        np.random.shuffle(folders)
        folders = folders[:n_classes]
        
        images = [glob.glob(os.path.join(directory, folder) + '/*jpg') for folder in folders]
        labels = [len(img_paths)*[folder] for folder, img_paths in zip(folders, images)]
        data = [pd.DataFrame({'filename': image, 'label': label}) for image, label in zip(images, labels)]
        df = pd.concat(data, ignore_index=True)
        
        
        if encoder is not None:
            df['class'] = encoder.fit_transform(df['label'])
            
        train, test = train_test_split(df, test_size=test_split, random_state=self.seed)
        
        train.to_csv(prefix + 'train.csv')
        test.to_csv(prefix + 'test.csv')
        
    def flow_from_files(self, db, x_col='filename', y_col='class', augmentor=None, val_split=0.2, batch_size=None, shuffle=True, random_state=None):
        
        df = pd.read_csv(db + '.csv')
        
        image_paths = df[x_col].values
        labels = df[y_col].values
            
        if random_state is not None:
            self.seed = random_state
        if batch_size is None:
            batch_size = self.batch_size
        
        if val_split > 0:
            X_train, X_test, Y_train, Y_test = train_test_split(image_paths, labels, test_size=val_split, random_state=self.seed)
            train_data = ImgDataGenerator(X_train, Y_train, batch_size=batch_size, augmentor=augmentor, shuffle=shuffle)
            val_data = ImgDataGenerator(X_test, Y_test, batch_size=batch_size, augmentor=None, shuffle=False)
            return train_data, val_data
        else:
            return ImgDataGenerator(image_paths, labels, batch_size=batch_size, augmentor=augmentor, shuffle=shuffle), None