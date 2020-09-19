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
from tensorflow.keras.preprocessing import image_dataset_from_directory


# load helper funcs
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), './'))
import helper_funcs as my_funcs


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

def load_data_from_dir(directory, batch_size, img_size, testsize):
    X_train = image_dataset_from_directory(
    directory, labels='inferred', class_names=None, 
    color_mode='rgb', batch_size=batch_size, image_size=(img_size, img_size), shuffle=True, seed=42,
    validation_split=0.25, subset='training', interpolation='bilinear', follow_links=True
    )
    # batch size needs to be hard coded to split for holdout
    testsize = 1968
    X_test = image_dataset_from_directory(
    directory, labels='inferred', class_names=None, 
    color_mode='rgb', batch_size=testsize, image_size=(img_size, img_size), shuffle=True, seed=42, 
    validation_split=0.25, subset='validation', interpolation='bilinear', follow_links=True
    )

    # calc sizes
    holdout_size = int(0.2 * testsize)
    test_size = testsize - holdout_size
    print(f' holdout size: {holdout_size}, test size: {test_size}')

    # pull X and y in tensors
    X_test_images, X_test_labels = next(iter(X_test))
    # split the first into holdout
    X_holdout_images = X_test_images[:holdout_size,...]
    X_holdout_labels = X_test_labels[:holdout_size]
    # put the rest in X_test
    X_test_images = X_test_images[holdout_size:,...]
    X_test_labels = X_test_labels[holdout_size:]
    # put into datasets
    X_test1 = tensorflow.data.Dataset.from_tensor_slices((X_test_images, X_test_labels))
    X_holdout1 = tensorflow.data.Dataset.from_tensor_slices((X_holdout_images, X_holdout_labels))

    return X_train, X_test1, X_holdout1

def prep_data(X_train, X_test, batch_size=None):
    if batch_size:
        X_test = X_test.batch(batch_size)
    X_train = X_train.cache().shuffle(32, seed=42).prefetch(buffer_size=AUTOTUNE) 
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)
    return X_train, X_test


def build_model(num_classes, nb_filters, kernel_size, pool_size, img_height, img_width, final_dense, model_dir=None):
    if model_dir:
        model = keras.models.load_model(model_dir)
        return model
    
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=img_height, img_width),
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                    input_shape=img_height, img_width),
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
    # raw data:
    directory = '/home/ec2-user/data/all_US_unaugmented'

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
    epochs = 10 
    AUTOTUNE = data.experimental.AUTOTUNE
    nb_filters = 32    
    pool_size = (2, 2)  
    kernel_size = (2, 2) 

    # run steps
    # X_train, X_test = load_datasets(X_train_data_path, X_test_data_path)
    # or with data not datasets
    X_train, X_test, X_holdout = load_data_from_dir(directory, batch_size, img_size)
    X_train, X_test = prep_data(X_train, X_test, batch_size)
    model = build_model()

    # check
    print(model.summary())

    my_callbacks = [
    callbacks.EarlyStopping(patience=25),
    #     callbacks.ModelCheckpoint(
    #                         filepath='../data/tensorboard_models/model.{epoch:02d}-{val_loss:.2f}.h5', 
    #                         monitor='val_loss', 
    #                         verbose=0, 
    #                         save_best_only=False,
    #                         save_weights_only=False, 
    #                         mode='auto', 
    #                         save_freq='epoch', 
    #                         options=None),
        callbacks.TensorBoard(log_dir='../tensorboard_logs',
                            histogram_freq=2,
                            write_graph=True,
                            write_images=True),
    ]

    # fit model
    history = model.fit(
            X_train,
            validation_data = X_test,
            epochs = epochs,
            verbose = 2,
#            callbacks=my_callbacks
)

    # name model
    model_name = '200_epochs_all_US_model_wild_est_binary'
    model_name = 'test_all_US_model_wild_est_binary'

    # save model
    model.save(f'../model_data/models/{model_name}')

    # save model perf plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    my_funcs.plot_train_val_acc(history, epochs, model_name, axs)
    plt.savefig(f'../model_data/plots/{model_name}_model_perf_accuracy.png')

    # save example images
    num_samples = 10
    figsize = (15,8)
    my_funcs.plot_example_imgs(X_test, class_names, figsize, num_samples)
    plt.savefig(f'../model_data/plots/{model_name}_example_imgs.png')

    # get and save conf mat and ROC
    y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras = my_funcs.run_model(model, X_test)
    # classification report
    class_report_dict = classification_report(y, y_pred_bin, output_dict=True)
    class_report_df = pd.DataFrame(class_report_dict).transpose()
    class_report_df.to_csv(f'../model_data/data/{model_name}_classification_report.csv')
    # ROC curve
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax = my_funcs.get_ROC_plot(ax, fpr_keras, tpr_keras, auc_keras, f'ROC curve - {model_name}')
    plt.savefig(f'../model_data/plots/{model_name}_ROC_curve.png')
    # confusion matrix
    confmat = my_funcs.compute_confusion_matrix(y, y_pred_bin)
    x_labels = ['Predict: Established', 'Predict: Wild'] 
    y_labels = ['Actual: Established', 'Actual: Wild']
    fig, ax = plt.subplots(1, figsize = (8,6))
    ax = my_funcs.plot_conf_matrix(confmat, ax, x_labels, y_labels, f'conf matrix for {model_name}')
    plt.savefig(f'../model_data/plots/{model_name}_conf_matrix.png')
    # output some incorrect predictions
    y_predictions_df = my_funcs.get_imgs_into_df(X_test, y, y_pred_bin)
    wrong_imgs = y_predictions_df[y_predictions_df['predict'] != y_predictions_df['actual']]
    num_samples = 10
    figsize = (20,8)
    fig, axs = my_funcs.plot_wrong_imgs(wrong_imgs, figsize, num_samples)
    plt.savefig(f'../model_data/plots/{model_name}_incorrect_predictions_sample.png')
