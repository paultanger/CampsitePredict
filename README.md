# Campsite classification and image analysis

## Overview

This project stems out of my interest and interest of many outdoor enthusiasts to find amazing 
and undiscovered areas to go camping. There are several resources currently to locate interesting 
campsites, but they depend on user entered campsite information and curation.

What if we could scan satellite imagery and predict where good campsites will be?

For a more detailed overview of this project, check the slides here:

https://paultanger.github.io/CampsitePredict/

I believe the techniques used here can be utilized for other purposes but initially the focus will 
be on using machine learning techniques with known campsites to predict the campsite category. 
Initially this project will use data available from the ioverlander mobile app to train and validate predictions.

## Table of Contents

* [How to use](#How-to-use)
* [Data sources](#data-sources)
* [Data cleaning and aggregation, and EDA](#Data-cleaning-and-aggregation,and-Exploratory-Data-Analysis)
* [Modeling](#Modeling)
* [Conclusions](#Conclusions)
* [Future plans](#future-plans)

## How to use

This project uses jupyter notebooks for the most part, with support functions in `src/helper_funcs.py`.
1. For now, the data has not been uploaded and shared but you can fetch it yourself from iOverlander and obtain images
using google Static Maps API with these notebooks (located in the notebooks directory): `get_states_zips.ipynb` and `get_images.ipynb`.  I first used reverse geocode to get State and Zip Code and then filtered the data to just keep sites (rows) with valid State information.
2. The training and validation data was examined in `summarize_data.ipynb` mostly to determine which States to start with.
3. The established campground data was examined and obtained in `established_campgrounds.ipynb` since I obtained that data separately.
4. The data and models were compiled and developed in `preprocess_and_build_model-binary.ipynb` and run on google colab with `run_sat_models_colab.ipynb`
5. The test data (from states the model had not seen before) was examined in `summarize_test_data.ipynb` to check class balances.
6. You can use `test_model_new_data.ipynb` notebook to load a saved model and test with new data and plot ROC, confusion matrix, example images, and examples of images where the model did not produce the correct classification.
7. The NLP analysis was completed in `NLP_of_descriptions.ipynb`.
8. Models that were explored but not utilized are in notebooks/model_exploration and src/archive.

## Data sources

Data files from iOverlander can be obtained here after filtering by country and place type: http://ioverlander.com/places/search.  Satelitte imagery was obtained using the Google Maps static API.

## Data cleaning, aggregation, and Exploratory Data Analysis

The cleaning and aggregation was completed using pandas in these notebooks and I briefly summarize here:
* summarize_data.ipynb
* summarize_test_data.ipynb

## Modeling

I created symlink directories of different sets of images to test different models, and several scripts just relate to that work (this layout might be unique to my workstation):
* organize_sat_data_only_unaugmented.ipynb
* organize_sat_data_dirs.ipynb
* sort_images.ipynb

Bringing together the original data set with the image predictions was accomplished with these scripts, and a modified version of the the image_dataset_from_directory returning also the image paths:

* summarize_image_filenames.ipynb
* summarize_All_US_data_clean.ipynb
* merge_df_and_images.ipynb

My original model was build using google colab from data from these states: XX
Upon obtaining additional data, my needs exceeded the RAM of colab, and I transitioned to building models on AWS EC2 P3 instances using these source files (in src):

* run_original500epoch_AWS.py

## Conclusions

I have been able to successfully use image classification neural networks to detect if a satellite image is a wild camping area or established campground.  In addition, I was able to use Natural Language Processing and K means clustering to split the Wild Camping category into two separate categories.  This could be useful for iOverlander to integrate into the app.

## Future plans

* I would like to add functionality to the flask app to accept input from users on whether they think the actual category label is correct based on the image and site description or if it should be re-labeled.  Then the model could take this information and improve (basically by having people curate the data set).
* I have not been able to find a data source that could provide if GPS coordinates are on public land or not.  I investigated https://www.usgs.gov/core-science-systems/science-analytics-and-synthesis/gap/science/protected-areas and there might be a way using ArcGIS but I did not see an API for this data.
* add flags for additional parameters the iOverlander team would like.
* While I was able to align the prediction data set with the original site information, I have not integrated this information into a model and I would like to use Keras functional API to provide this as additional inputs.  Some vey fields could be the Water, and Bathrooms fields (as binary). 
* Implement object detection approaches to pinpoint the predicted campsite location within the satellite image and obtain the GPS coordinates for these predictions.