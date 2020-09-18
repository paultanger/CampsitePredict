import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

import os

from tensorflow import TensorSpec, float32, int32
from tensorflow import data
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.data.experimental import load
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks

# load helper funcs
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'/content/drive/My Drive/src'))
# import helper_funcs as my_funcs


def load_datasets(X_train_file, X_test_file):
    # set element spec
    X_train_elem_spec = (TensorSpec(shape=(None, img_height, img_width, 3), dtype=float32, name=None), 
                        TensorSpec(shape=(None,), dtype=int32, name=None))
    X_test_elem_spec = (TensorSpec(shape=(img_height, img_width, 3), dtype=float32, name=None), 
                        TensorSpec(shape=(), dtype=int32, name=None))

    # get files
    X_train = load(X_train_file, element_spec=X_train_elem_spec, compression='GZIP', reader_func=None)
    X_test = load(X_test_file, element_spec=X_test_elem_spec, compression='GZIP', reader_func=None)
    return X_train, X_test


def prep_data(X_train, X_test, batch_size):
    X_test = X_test.batch(batch_size)
    X_train = X_train.cache().shuffle(32, seed=42).prefetch(buffer_size=AUTOTUNE) 
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)
    return X_train, X_test


def build_model(model_dir=None):
    if model_dir:
        model = keras.models.load_model(model_dir)
        return model
    
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                    input_shape=(img_height, 
                                                                img_width,
                                                                3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same', activation='relu'), # was 16, 32, 64
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(nb_filters*2, (kernel_size[0], kernel_size[1]), padding='same', activation='relu'), # drop layers.. for initial testing
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(nb_filters*3, (kernel_size[0], kernel_size[1]), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Conv2D(nb_filters*4, (kernel_size[0], kernel_size[1]), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=pool_size),
        layers.Flatten(),
        layers.Dense(final_dense, activation='relu'),
        layers.Dropout(0.5),
        #layers.Dense(num_classes, activation='relu')
        layers.Dense(1, activation='sigmoid')
        ])

    model.compile(optimizer='adam', # adadelta sgd
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # aws s3 cp s3://my_bucket/my_folder/my_file.ext my_copied_file.ext
    # aws s3 cp s3://campsite-data/data data --recursive
    # os.system('source activate tensorflow2_latest_p37')
    print(keras.__version__)
    print(tensorflow.__version__)
    # os.system('nvidia-smi')

    # path to files:
    X_train_data_path = '/home/ec2-user/data/all_US_data/X_train_256px_32batch'
    X_test_data_path = '/home/ec2-user/data/all_US_data/X_test_256px_unbatched'

    # get class names for plotting
    class_names = ['Est Camp', 'Wild Camp']

    # 
    batch_size = 32
    img_size = 256
    img_height = 256
    img_width = 256
    final_dense = 256

    # set params
    num_classes = 2
    epochs = 200 
    AUTOTUNE = data.experimental.AUTOTUNE
    nb_filters = 32    
    pool_size = (2, 2)  
    kernel_size = (2, 2) 

    # run steps
    X_train, X_test = load_datasets(X_train_data_path, X_test_data_path)
    X_train, X_test = prep_data(X_train, X_test, batch_size)
    model = build_model()

    # check
    print(model.summary())

    my_callbacks = [
    callbacks.EarlyStopping(patience=5),
    #     callbacks.ModelCheckpoint(
    #                         filepath='../data/tensorboard_models/model.{epoch:02d}-{val_loss:.2f}.h5', 
    #                         monitor='val_loss', 
    #                         verbose=0, 
    #                         save_best_only=False,
    #                         save_weights_only=False, 
    #                         mode='auto', 
    #                         save_freq='epoch', 
    #                         options=None),
        callbacks.TensorBoard(log_dir='./tensorboard_logs',
                            histogram_freq=2,
                            write_graph=True,
                            write_images=True),
    ]

    # fit model
    history = model.fit(
            X_train,
            validation_data = X_test,
            epochs = epochs,
            verbose = 2
            callbacks=my_callbacks
)

    # save model
    model.save('./models/200_epochs_model_wild_est_binary')
