import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras

import os
import sys

from tensorflow import TensorSpec, float32, int32
from tensorflow import data
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.data.experimental import load
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import optimizers

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
    directory, labels='inferred', class_names=None, label_mode='categorical',
    color_mode='rgb', batch_size=batch_size, image_size=(img_size, img_size), shuffle=True, seed=42,
    validation_split=0.25, subset='training', interpolation='bilinear', follow_links=True
    )
    # batch size needs to be hard coded to split for holdout
    # testsize = 1968
    X_test = image_dataset_from_directory(
    directory, labels='inferred', class_names=None, label_mode='categorical',
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

def prep_data(X_train, X_test, batch_size=None, buffer_size=data.experimental.AUTOTUNE):
    if batch_size:
        X_test = X_test.batch(batch_size)
    X_train = X_train.cache().shuffle(32, seed=42).prefetch(buffer_size=buffer_size) 
    X_test = X_test.cache().prefetch(buffer_size=buffer_size)
    return X_train, X_test

def get_class_weights(X_train):
    class_names = X_train.class_names
    labels = np.concatenate([y for x, y in X_train], axis=0)
    if np.ndim(labels) == 1:
        # for binary
        weights = [len(labels) - labels.sum(), labels.sum()]
    else:
        weights = list(np.sum(labels, axis=0))
    class_weights = {}
    # for class_, weight in zip(class_names, weights):
    #     class_weights[class_] = weight
    for i, weight in enumerate(weights):
        class_weights[i] = weight
    return class_names, class_weights

def build_model(num_classes, nb_filters, kernel_size, pool_size, img_height, img_width, final_dense, model_dir=None):
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
        #layers.Dropout(0.3),
        # layers.Dense(128, activation='softmax'),
        layers.Dense(num_classes, activation='softmax')
        # layers.Dense(1, activation='sigmoid')
        ])

    slow_adam = optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=slow_adam, #'adam', # adadelta sgd
            #   loss=keras.losses.BinaryCrossentropy(from_logits=False),
            #   metrics=['accuracy'])
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            # metrics=['accuracy'])
            metrics=['accuracy', 'Recall'])

    return model

def get_preds(model, X_test):
    '''
    requires numpy as np and sklearn metrics
    gets model predictions from keras model
    returns: y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras,
    thresholds_keras, auc_keras
    '''
    predictions = model.predict(X_test, verbose=2)
    # y_pred_bin = (predictions > 0.5).astype("int32")
    y = np.concatenate([y for x, y in X_test], axis=0)
    y_pred = predictions.ravel()
    # fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y, y_pred)
    # auc_keras = metrics.auc(fpr_keras, tpr_keras)
    return y, predictions, y_pred, #y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras

def multiclass_ROC_plot(class_names, y_test, y_score, ax, title):
    # Compute ROC curve and ROC area for each class
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    #     plt.figure()
    #     plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    #     plt.plot([0, 1], [0, 1], 'k--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(title)
    #     plt.legend(loc="lower right")
    #     plt.show()

        # Plot ROC curve
    #     plt.figure()
    ax.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(class_names[i], roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    return ax

if __name__ == "__main__":
    # aws s3 cp s3://my_bucket/my_folder/my_file.ext my_copied_file.ext
    # aws s3 cp s3://campsite-data/data data --recursive
    # os.system('source activate tensorflow2_latest_p37')
    model_name = sys.argv[1]
    directory = sys.argv[2]
    test_data_size = int(sys.argv[3])
    epochs = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    print(epochs)
    print(f'model name: {model_name} \n dir: {directory}')
    print(model_name)
    print(tensorflow.__version__)
    # raw data:
    # directory = '/home/ec2-user/data/all_US_unaugmented'

    # 
    # batch_size = 32
    img_size = 256
    img_height = 256
    img_width = 256
    final_dense = 256

    # run steps
    # X_train, X_test = load_datasets(X_train_data_path, X_test_data_path)
    # or with data not datasets
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
    output_bias = np.log(bias[0] / bias[1])
    output_bias = np.array([np.log(bias[0] / bias[1])])
    
    X_train, X_test = prep_data(X_train, X_test, batch_size=batch_size)
    model = build_model(num_classes, nb_filters, kernel_size, pool_size, img_height, img_width, final_dense)

    # check
    print(model.summary())

    # fit model
    history = model.fit(
            X_train,
            validation_data = X_test,
            epochs = epochs,
            class_weight = class_weights,
            verbose = 2,
)

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
    y, predictions, y_pred = get_preds(model, X_test)
    # classification report
    # class_report_dict = classification_report(y, y_pred_bin, output_dict=True)
    # class_report_df = pd.DataFrame(class_report_dict).transpose()
    # class_report_df.to_csv(f'../model_data/data/{model_name}_classification_report.csv')
    # ROC curve
    fig, ax = plt.subplots(1, figsize=(10, 8))
    multiclass_ROC_plot(class_names, y, predictions, ax, f'multi-class ROC for {model_name}')
    plt.savefig(f'../model_data/plots/{model_name}_ROC_curve.png')
    # confusion matrix
    confmat = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1), normalize='all')
    xlabels = [f'actual: {x}' for x in class_names] 
    ylabels = [f'pred: {x}' for x in class_names] 
    fig, ax = plt.subplots(1, figsize = (8,6))
    ax = my_funcs.plot_conf_matrix(confmat, ax, xlabels, ylabels, f'conf matrix for {model_name}')
    plt.savefig(f'../model_data/plots/{model_name}_conf_matrix.png')