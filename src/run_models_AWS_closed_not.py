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
from  run_multiclass_models_AWS import prep_data, get_class_weights
from  run_models_AWS import load_data_from_dir, build_model_imb


if __name__ == "__main__":
    print(tensorflow.__version__)
    model_name = sys.argv[1]
    directory = sys.argv[2]
    # test_data_size = int(sys.argv[3]) 1177
    test_data_size = 1177
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])

    # raw data:
    directory = '/home/ec2-user/data/wild_closed_not'

    img_size = 256
    img_height = 256
    img_width = 256
    final_dense = 256

    X_train, X_test, X_holdout = load_data_from_dir(directory, batch_size, img_size, test_data_size)
    # get class names for plotting and weights
    class_names, class_weights = get_class_weights(X_train)
    # set params
    num_classes = len(X_train.class_names)
    # epochs = 10 
    AUTOTUNE = data.experimental.AUTOTUNE
    nb_filters = 32    
    pool_size = (2, 2)  
    kernel_size = (2, 2) 

    # calc bias
    bias = np.array(list(class_weights.values())) 
    output_bias = np.array([np.log(bias[0] / bias[1])])

    X_train, X_test = prep_data(X_train, X_test, batch_size)
    model = build_model_imb(num_classes, nb_filters, kernel_size, pool_size, img_height, img_width, final_dense, output_bias)

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
            class_weight = class_weights
#            callbacks=my_callbacks
)

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
    xlabels = [f'actual: {x}' for x in class_names] 
    ylabels = [f'pred: {x}' for x in class_names] 
    fig, ax = plt.subplots(1, figsize = (8,6))
    ax = my_funcs.plot_conf_matrix(confmat, ax, xlabels, ylabels, f'conf matrix for {model_name}')
    plt.savefig(f'../model_data/plots/{model_name}_conf_matrix.png')
    # output some incorrect predictions
    y_predictions_df = my_funcs.get_imgs_into_df(X_test, y, y_pred_bin)
    wrong_imgs = y_predictions_df[y_predictions_df['predict'] != y_predictions_df['actual']]
    num_samples = 10
    figsize = (20,8)
    fig, axs = my_funcs.plot_wrong_imgs(wrong_imgs, figsize, num_samples)
    plt.savefig(f'../model_data/plots/{model_name}_incorrect_predictions_sample.png')
