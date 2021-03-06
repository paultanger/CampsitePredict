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
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), './'))
import helper_funcs as my_funcs
from image_dataset_from_dir_return_paths import image_dataset_from_directory_paths

def load_data_from_dir(directory, batch_size, img_size, testsize, shuffle_test=True):
    X_train, X_train_img_paths = image_dataset_from_directory_paths(
    directory, labels='inferred', class_names=None, 
    color_mode='rgb', batch_size=batch_size, image_size=(img_size, img_size), shuffle=True, seed=42,
    validation_split=0.25, subset='training', interpolation='bilinear', follow_links=True
    )
    # batch size needs to be hard coded to split for holdout
    # testsize = 1968
    if not shuffle_test:
        shuffle_test = False
    X_test, X_test_img_paths = image_dataset_from_directory_paths(
    directory, labels='inferred', class_names=None, 
    color_mode='rgb', batch_size=testsize, image_size=(img_size, img_size), shuffle=shuffle_test, seed=42, 
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
    X_holdout1_img_paths = X_test_img_paths[:holdout_size]
    # put the rest in X_test
    X_test_images = X_test_images[holdout_size:,...]
    X_test_labels = X_test_labels[holdout_size:]
    X_test_img_paths = X_test_img_paths[holdout_size:]
    # put into datasets
    X_test1 = tensorflow.data.Dataset.from_tensor_slices((X_test_images, X_test_labels))
    X_holdout1 = tensorflow.data.Dataset.from_tensor_slices((X_holdout_images, X_holdout_labels))

    return X_train, X_test1, X_holdout1, X_train_img_paths, X_test_img_paths, X_holdout1_img_paths

def load_data_from_dir_all_for_predictions(directory, batch_size, img_size, testsize, shuffle_test=True):
    if not shuffle_test:
        shuffle_test = False
    X_test, X_test_img_paths = image_dataset_from_directory_paths(
    directory, labels='inferred', class_names=None, 
    color_mode='rgb', batch_size=testsize, image_size=(img_size, img_size), shuffle=shuffle_test, seed=42, 
    interpolation='bilinear', follow_links=True
    )

    return X_test, X_test_img_paths

def prep_data(X_test, batch_size=None):
    if batch_size:
        X_test = X_test.batch(batch_size)
    # X_train = X_train.cache().shuffle(32, seed=42).prefetch(buffer_size=AUTOTUNE) 
    X_test = X_test.cache().prefetch(buffer_size=AUTOTUNE)
    return X_test


if __name__ == "__main__":
    print(tensorflow.__version__)
    # model_name = sys.argv[1]
    # directory = sys.argv[2]
    # test_data_size = int(sys.argv[3])
    # epochs = int(sys.argv[4])
    # batch_size = int(sys.argv[5])
    directory = '/Users/pault/Desktop/github/CampsitePredict/data/symlink_data/unique_wild_est_for_aligned_model'
    model_name = 'test_500_epoch_all_US_nodups_all7834'
    test_data_size = 7834
    batch_size = 32
    img_size = 256
    img_height = 256
    img_width = 256
    AUTOTUNE = data.experimental.AUTOTUNE
    # load data
    X_test, X_test_img_paths = load_data_from_dir_all_for_predictions(directory, batch_size, img_size, test_data_size, shuffle_test=False)
    class_names, class_weights = my_funcs.get_class_weights(X_test)
    # run steps
    # since it is already being batched don't need
    X_test = prep_data(X_test)

    # load model to run predictions
    model = keras.models.load_model('../data/models/500_epochs_model_wild_est_binary')

    # check
    print(model.summary())

    # # run model
    y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras = my_funcs.run_model(model, X_test)


    # test image - should be predicting correctly as wild (1)..
    path = '../../media/images/satimg_AK_10199_Wild Camping_17_62.724088_-141.181026.png'
    path = '../../media/images/satimg_AK_1028_Established Campground_17_60.958322_-149.111734.png'
    path = '../web_app/static/temp_images/satimg_17_38.497862_-106.945057.png'
    # this should be wild:
    path = '../web_app/static/temp_images/satimg_17_62.724088_-141.181026.png'
    cropped = plt.imread(path)
    plt.imshow(cropped)
    plt.show()
    # prep for model
    from skimage.transform import resize
    cropped = cropped[:,:,:3]
    cropped = resize(cropped, (256, 256))
    # drop alpha
    # cropped = cropped[:,:,:3]
    test_img = np.expand_dims(cropped, axis=0) 
    # make model prediction
    predictions = model.predict(test_img)
    y_pred_bin = (predictions > 0.5).astype("int32")
    if y_pred_bin == 1:
        predict_text = 'Wild Camping' 
    else:
        predict_text = 'Established Campground'
    path = '/Users/pault/Desktop/github/CampsitePredict/web_app/static/temp_images/test'
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    path = '../web_app/static/temp_images/satimg_17_62.724088_-141.181026.png'
    cropped = plt.imread(path)
    plt.imshow(cropped)
    plt.show()
    testimg2 = load_img(path, grayscale=False, color_mode="rgb", target_size=(256,256), interpolation="bilinear")
    input_arr = keras.preprocessing.image.img_to_array(testimg2)
    input_arr = np.array([input_arr]) 
    model.predict(input_arr)
    # will pil also work?
    from PIL import Image
    im = Image.open(path)
    im = im.resize((256, 256), resample=Image.BILINEAR) 
    input_arr = keras.preprocessing.image.img_to_array(im)
    input_arr_rgb = input_arr[:,:,:3]
    input_arr_rgb = np.array([input_arr_rgb])
    model.predict(input_arr_rgb)
    X_test, X_test_img_paths = load_data_from_dir_all_for_predictions(path, 32, 256, 32, shuffle_test=False)


    # # get predictions together with image filename and original df with other data
    # y_predictions_df = my_funcs.get_imgs_into_df(X_test, y, y_pred_bin)
    # # y_predictions_df.to_csv('../data/test.csv')
    # # add correct col
    # y_predictions_df['correct'] = 0
    # y_predictions_df['correct'][y_predictions_df['predict'] == y_predictions_df['actual']] = 1
    # # add the filenames
    # y_predictions_df['filepaths'] = X_test_img_paths
    # y_predictions_df['filename'] = pd.Series(y_predictions_df['filepaths'].str.rsplit('/', n=1, expand=True)[1])
    # # merge with original df
    # original_df = pd.read_csv('../data/image_file_df_final_with_df_NO_DUPS.csv')
    # original_df.drop('Unnamed: 0', axis=1, inplace=True)
    # # add png so they will match
    # original_df['filename'] = original_df['filename'] + '.png'
    # # merge
    # df_with_preds = y_predictions_df.merge(original_df, how='left', on = 'filename')
    # # save it
    # df_with_preds.to_csv('../data/df_with_preds3.tsv', sep='\t')
    # # drop the actual image arrays and merge
    # y_predictions_df_no_imgs = y_predictions_df.drop('image', axis=1)
    # df_with_preds_no_imgs = y_predictions_df_no_imgs.merge(original_df, how='left', on = 'filename')
    # # save it
    # df_with_preds_no_imgs.to_csv('../data/df_with_preds_no_imgs3.tsv', sep='\t')

    # test = df_with_preds.iloc[0,2]
    # plt.imshow(test) 
    # plt.show()
    # # save example images
    # num_samples = 10
    # figsize = (15,8)
    # my_funcs.plot_example_imgs(X_test, class_names, figsize, num_samples)
    # plt.savefig(f'../model_data/plots/{model_name}_example_imgs.png')

    # # get and save conf mat and ROC
    # # y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras = my_funcs.run_model(model, X_test)
    # # classification report
    # class_report_dict = classification_report(y, y_pred_bin, output_dict=True)
    # class_report_df = pd.DataFrame(class_report_dict).transpose()
    # class_report_df.to_csv(f'../model_data/data/{model_name}_classification_report.csv')
    # # ROC curve
    # fig, ax = plt.subplots(1, figsize=(10, 8))
    # ax = my_funcs.get_ROC_plot(ax, fpr_keras, tpr_keras, auc_keras, f'ROC curve - {model_name}')
    # plt.savefig(f'../model_data/plots/{model_name}_ROC_curve.png')
    # # confusion matrix
    # confmat = my_funcs.compute_confusion_matrix(y, y_pred_bin)
    # xlabels = [f'actual: {x}' for x in class_names] 
    # ylabels = [f'pred: {x}' for x in class_names] 
    # fig, ax = plt.subplots(1, figsize = (8,6))
    # ax = my_funcs.plot_conf_matrix(confmat, ax, xlabels, ylabels, f'conf matrix for {model_name}')
    # plt.savefig(f'../model_data/plots/{model_name}_conf_matrix.png')
    # # output some incorrect predictions
    # y_predictions_df = my_funcs.get_imgs_into_df(X_test, y, y_pred_bin)
    # wrong_imgs = y_predictions_df[y_predictions_df['predict'] != y_predictions_df['actual']]
    # num_samples = 10
    # figsize = (20,8)
    # fig, axs = my_funcs.plot_wrong_imgs(wrong_imgs, figsize, num_samples)
    # plt.savefig(f'../model_data/plots/{model_name}_incorrect_predictions_sample.png')
