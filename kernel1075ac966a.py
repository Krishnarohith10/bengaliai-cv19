import gc
import cv2
import tensorflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Input, AveragePooling2D, Dropout, Flatten

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
class_map = pd.read_csv('data/class_map.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

train = train.drop(['grapheme'], axis=1)
test = test.drop(['row_id', 'component'], axis=1)

for i in tqdm(range(4)):
    train_image_data = pd.read_parquet('data/train_image_data_{}.parquet'.format(i), engine='pyarrow')
    train_df = pd.merge(train_image_data, train, on='image_id').drop(['image_id'], axis=1)

del train_image_data

def resize(df):
    resized = {}
    resize_size=64
    for i in tqdm(range(df.shape[0])):
        img = df.iloc[i, :].values.reshape(137, 236)
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        resized[i] = img.reshape(-1)
    resized_df = pd.DataFrame(resized).T
    return resized_df

SIZE=64
CHANNELS=1
x_train = resize(train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1))/255.0
x_train = x_train.values.reshape(-1, SIZE, SIZE, CHANNELS)

y_train_grapheme = pd.get_dummies(train_df['grapheme_root']).values
y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values

del train
del train_df
del class_map
del sample_submission

batch_size = 32
epochs = 50
n = 3
depth = n * 9 + 2
input_shape = x_train.shape[1:]

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth):

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, 
                             activation=activation, batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)
            
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1,
                                 strides=strides, activation=None, batch_normalization=False)
            
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    output_grapheme = Dense(168, activation='softmax', 
                            kernel_initializer='he_normal', name='output_grapheme')(y)
    output_vowel = Dense(11, activation='softmax', 
                         kernel_initializer='he_normal', name='output_vowel')(y)
    output_consonant = Dense(7, activation='softmax', 
                             kernel_initializer='he_normal', name='output_consonant')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[output_grapheme, output_vowel, output_consonant])
    # Compile for learning
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = resnet_v2(input_shape=input_shape, depth=depth)

model.summary()

lr_reducer_grapheme = ReduceLROnPlateau(monitor='output_grapheme_accuracy', 
                                    factor=np.sqrt(0.1),
                                    patience=5, 
                                    verbose=1,
                                    min_lr=0.5e-6)
lr_reducer_vowel = ReduceLROnPlateau(monitor='output_vowel_accuracy', 
                                     factor=np.sqrt(0.1),
                                     patience=5, 
                                     verbose=1,
                                     min_lr=0.5e-6)
lr_reducer_consonant = ReduceLROnPlateau(monitor='output_consonant_accuracy', 
                                         factor=np.sqrt(0.1),
                                         patience=5, 
                                         verbose=1, 
                                         min_lr=0.5e-6)

callbacks = [lr_reducer_grapheme, lr_reducer_vowel, lr_reducer_consonant]
'''
print('Using real-time data augmentation.')
# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
        featurewise_center=False, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        zca_epsilon=1e-06, # epsilon for ZCA whitening
        rotation_range=0, # randomly rotate images in the range (deg 0 to 180)
        width_shift_range=0.1, # randomly shift images horizontally
        height_shift_range=0.1, # randomly shift images vertically
        shear_range=0., # set range for random shear
        zoom_range=0., # set range for random zoom
        channel_shift_range=0., # set range for random channel shifts
        fill_mode='nearest', # set mode for filling points outside the input boundaries
        cval=0., # value used for fill_mode = "constant"
        horizontal_flip=True, # randomly flip images
        vertical_flip=False, # randomly flip images
        rescale=None, # set rescaling factor (applied before any other transformation)
        preprocessing_function=None, # set function that will be applied on each input
        data_format=None, # image data format, either "channels_first" or "channels_last"
        validation_split=0.08) # fraction of images reserved for validation (strictly between 0 and 1)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(datagen.flow(x=x_train, 
                                           y={'output_grapheme': y_train_grapheme, 
                                              'output_vowel': y_train_vowel, 
                                              'output_consonant': y_train_consonant}, 
                                           batch_size=batch_size), 
                              epochs=epochs, verbose=1, 
                              steps_per_epoch = x_train.shape[0]//batch_size, 
                              callbacks=callbacks)
'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, {'output_grapheme': y_train_grapheme, 
                              'output_consonant': y_train_consonant, 
                              'output_vowel': y_train_vowel}, batch_size=batch_size, 
                    epochs=epochs, verbose=1, callbacks=callbacks)

del x_train
del y_train_grapheme
del y_train_vowel
del y_train_consonant
gc.collect()

model.save('bengali-cv19-model.h5')
