#!/usr/bin/env python3
# author: Paul Tanger
# date modified: 

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from random import choices
from sklearn.cluster import KMeans
from collections import Counter

def nice_filename(fname, extension):
    '''
    takes filename and extension and returns nice formatted name
    '''
    FORMAT = '%Y%m%d_%H%M'
    return fname + '_' + datetime.now().strftime(FORMAT) + '.' + extension

def get_state_zip(df):
    '''
    gets zip code and state and return as two separate lists (zips, states)
    '''
    # TODO: add verbosity options
    zips = []
    states = []
    
    global n_requests
    
    for i, site in df.iterrows():
        result = None
        print(f'Pulling request {i}, total API requests so far = {n_requests}')
        
        # get latlong in right format
        # temp save coords
        lat = str(site['Latitude'])
        long = str(site['Longitude'])
        latlong = lat + ',' + long
        # get geocode data
        res_type = 'postal_code' # could also get: administrative_area_level_1|
        result = gmaps.reverse_geocode(latlong, result_type=res_type)
        # increase counter
        n_requests += 1
        if result:
            # pull out things we need
            zip_code = result[0]['address_components'][0]['short_name']
            # sometimes the index isn't the same
            #state = result[0]['address_components'][3]['short_name']
            types = ['administrative_area_level_1', 'political']
            statethingy = [d.items() for d in result[0]['address_components'] if d['types'] == types]
            # sometimes there isn't this type (like Puerto Rico)
            if statethingy:
                state = [x[1] for x in statethingy[0] if x[0] == 'short_name'][0]
            else:
                # if it didn't work for this row
                zip_code = ''
                state = ''
        else:
            # if it didn't work for this row
            zip_code = ''
            state = ''
        
        # append
        zips.append(zip_code)
        states.append(state)
        
        # wait a bit before next request
        wait_time = random.randint(1, 3)
        print(f'waiting for: {wait_time} seconds')
        time.sleep(wait_time) # in seconds
    
    return zips, states

def download_images(client, df, zoomlevel, max_requests=10, prefix="", out_path="../data/"):
    '''
    downloads satellite images using a google API, a df with a Latitude and Longitude column and a Category column
    You can specify zoom level, and where to save the images.
    You can also specify a prefix for the image names.
    This requires matplotlib.pyplot as plt, pandas as pd and os.
    Returns True when it is done.
    '''
    # TODO: add options for verbosity
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    global n_requests
    
    for i, site in df.iterrows():
        print(f'Pulling image {i}, total API requests so far = {n_requests}')
        
        # temp save coords
        lat = site['Latitude']
        long = site['Longitude']
        
        # and tags for site
        cat = site['Category']
        
        # create filename
        cur_filename = f'satimg_{prefix}_{i}_{cat}_{zoomlevel}_{lat}_{long}.png'
        print(cur_filename)

        # if it already exists, skip to next
        if os.path.exists(out_path + cur_filename):
            continue

        # get the image
        satimg = client.static_map(size = (400, 400), # pixels
           zoom = zoomlevel, # 1-21
           center = (lat, long),
           scale = 1, # default is 1, 2 returns 2x pixels for high res displays
           maptype = "satellite",
           format = "png"
          )
        
        # if it didn't work, exit
        if satimg is None or n_requests >= max_requests:
            print("API requests quota exceeded!")    
            break
        # increase counter otherwise
        n_requests += 1
        
        # save the current image
        f = open(out_path + cur_filename, 'wb')
        for chunk in satimg:
            if chunk:
                f.write(chunk)
        f.close()
        
        # open it to crop the text off
        img = plt.imread(out_path + cur_filename)
        # maybe crop all 4 sides?
        cropped = img[25:375, 25:375]
        # and resave
        plt.imsave(out_path + cur_filename, cropped)
        
        # and rotate and save each version
        for k, degrees in enumerate([90, 180, 270]):
            cropped_rotated = np.rot90(cropped, k=k)
            cropped_rot_filename = f'satimg_{prefix}_{i}_{cat}_{zoomlevel}_{lat}_{long}_rot{degrees}.png'
            plt.imsave(out_path + cropped_rot_filename, cropped_rotated)
        
        # wait a bit before next request
        wait_time = random.randint(1, 5)
        print(f'waiting for: {wait_time} seconds')
        time.sleep(wait_time) # in seconds

        # display samples every now and then
        if i % 100 == 0:
            img = plt.imread(out_path + cur_filename)
            plt.imshow(img)
            plt.title(f'image {i}')
            plt.show()
            time.sleep(2)
    return True

def run_model(model, X_test):
    '''
    requires numpy as np and sklearn metrics
    gets model predictions from keras model
    returns: y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras
    '''
    predictions = model.predict(X_test, verbose=2)
    y_pred_bin = (predictions > 0.5).astype("int32")
    y = np.concatenate([y for x, y in X_test], axis=0)
    y_pred = predictions.ravel()
    fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y, y_pred)
    auc_keras = metrics.auc(fpr_keras, tpr_keras)
    return y, predictions, y_pred, y_pred_bin, fpr_keras, tpr_keras, thresholds_keras, auc_keras

def compute_confusion_matrix(y_true_class, y_pred_class):
    '''
    computes confusion matrix and returns as array
    requires sklearn confusion_matrix
    '''
    confmat = confusion_matrix(y_true_class, y_pred_class, labels=range(2))
    confmat = confmat / confmat.sum(0).astype(float)
    return confmat

def plot_conf_matrix(confmat, ax, x_labels, y_labels, title):
    '''
    return ax with confusion matrix plot
    requires pandas as pd and seaborn as sns and matplotlib.pyplot as plt
    '''
    df_cm = pd.DataFrame(confmat, index = x_labels, columns = y_labels)
    sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    plt.yticks(va="center")
    plt.title(title)
    return ax

def get_ROC_plot(ax, fpr_keras, tpr_keras, auc_keras, title):
    '''
    return ax with ROC plot
    requires matplotlib.pyplot as plt
    '''
    ax.plot(fpr_keras, tpr_keras, label='model (area = {:.3f})'.format(auc_keras))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(title)
    plt.legend(loc='best')
    return ax

def get_imgs_into_df(X_test, y, predictions_binary_vec):
    '''
    accepts tf dataset and y true and binary predictions
    returns a df with predictions, actual, and images in cols
    '''
    # get images into df
    x_test_images = np.concatenate([x for x, y in X_test], axis=0)
    images = pd.Series(list(x_test_images))
    # get predictions into df
    ys = np.column_stack((predictions_binary_vec, y))
    y_predictions_df = pd.DataFrame(ys, columns=['predict', 'actual'])
    # replace labels
    y_predictions_df = y_predictions_df.replace(to_replace=0, value="est_camp") #['Established Campground', 'Wild Camping']
    y_predictions_df = y_predictions_df.replace(to_replace=1, value="wild_camp") #['Established Campground', 'Wild Camping']
    # add images col
    y_predictions_df['image'] = images
    return y_predictions_df

def plot_wrong_imgs(wrong_imgs, figsize=(15,15), num_samples=20):
    '''
    accepts a df of images (as numpy arrays) with the following cols:
    predict, actual, image (such as obtained from get_imgs_into_df())
    plots a grid with the predictions and actual images
    returns fig, axs
    '''
    # randomly pick images
    samples = choices(range(len(wrong_imgs)), k=num_samples)
    # setup figure
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['figure.titlesize'] = 'medium'
    fig, axs = plt.subplots(int(num_samples/4), 4, figsize=figsize)

    for i, ax in zip(samples, axs.flatten()):
        img = wrong_imgs.iloc[i]['image'].astype("uint8")
        pred = wrong_imgs.iloc[i]['predict']
        act = wrong_imgs.iloc[i]['actual']
        ax.imshow(img)
        ax.set_title(f'pred: {pred}, act: {act}')
        ax.axis('off')
    plt.tight_layout()
    return fig, axs

def plot_example_imgs(X_test, figsize=(15,15), num_samples=20):
    '''
    accepts a tf dataset of images
    plots a grid with sample images
    returns fig, axs
    '''
    # gets images out
    #x_test_images = np.concatenate([x for x, y in X_test], axis=0)
    image_batch, labels_batch = next(iter(X_test))
    # randomly pick images
    samples = choices(range(len(image_batch)), k=num_samples)
    # setup figure
    fig, axs = plt.subplots(int(num_samples/4), 4, figsize=figsize)

    for i, ax in zip(samples, axs.flatten()):
        img = image_batch[i].numpy().astype("uint8")
        label = labels_batch[i]
        if label:
            label = X_test.class_names[1]
        else:
            label = X_test.class_names[0]
        ax.imshow(img)
        ax.set_title(f'label: {label}')
        ax.axis('off')
    plt.tight_layout()
    return fig, axs

def plot_train_val_acc(history, epochs, ax):
    '''
    accepts model run history and number of epochs and fig ax
    plots train and val accuracy over epoch
    '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    ax.plot(epochs_range, acc, label='Training Accuracy')
    ax.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    ax.set_xlabel('epochs')
    ax.set_title('Training and Validation Accuracy')

    ax.plot(epochs_range, loss, label='Training Loss')
    ax.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    ax.set_xlabel('epochs')
    ax.set_title('Training and Validation Loss')
    return ax

def run_kmeans(X, df, features, k):
    '''
    accepts a TFIDF object (X), original df, features, and k (int)
    runs k means, gets top 20 cluster features, calcs % most common in each
    returns these as two dictionaries
    '''
    kmeans = KMeans(k)
    kmeans.fit(X)
    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-21:-1]
    cluster_feats = {}
    for num, centroid in enumerate(top_centroids):
        cluster_feats[num] = ', '.join(features[i] for i in centroid)

    assigned_cluster = kmeans.fit_transform(X).argmin(axis=1) # get the cluster assigned to each row (site)

    # save in dict
    cluster_cats = {}

    for i in range(kmeans.n_clusters):
        cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
        categories = df.iloc[cluster]['Category']
        most_common = Counter(categories).most_common()
        cluster_cats[i] = {}
        for j in range (len(most_common)):
            cluster_cats[i].update({most_common[j][0] : most_common[j][1]})
    return cluster_cats, cluster_feats

def get_cat_summary(cat_dict, cluster_feats, cluster_names=[]):
    '''
    accepts cat_dict, cluster_feats as obtained from run_kmeans()
    cluster_names is your guess as to the names of these
    returns category df and the max of each category
    '''
    cat_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in cat_dict.items()}, axis=0).reset_index()
    cat_df.columns = ['cluster', 'category', 'count']
    cat_df['cluster'] = cat_df['cluster'].astype('str')
    cat_df['pct_total'] = round(cat_df['count'].div(cat_df.groupby('cluster')['count'].transform('sum'))*100, 2)
    max_indices = cat_df.groupby(['cluster'])['pct_total'].transform(max) == cat_df['pct_total']
    cat_max = cat_df[max_indices].copy()
    cat_max['top words'] = pd.Series(cluster_feats).values
    cat_max['cluster name'] = cluster_names 
    return cat_max, cat_df