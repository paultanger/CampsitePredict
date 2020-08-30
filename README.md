# Campsite classification and image analysis

## Overview

This project stems out of my interest and interest of many outdoor enthusiasts to find amazing 
and undiscovered areas to go camping.  There are several resources currently to locate interesting 
campsites, but they depend on user entered campsite information and curation.

What if we could scan satellite imagery and predict where good campsites will be?

For a more detailed overview of this project, check the slides here:

https://paultanger.github.io/CampsitePredict/

I believe the techniques used here can be utilized for other purposes but initially the focus will 
be on using machine learning techniques with known campsites to characterize the satellite features 
of “good” campsites.  In addition, the slope, road access, proximity to buildings and other factors 
like is it located on public land (National Forest vs BLM land – which have different regulations) 
can be used to deliver the best undiscovered campsites to users.  Initially this project will use data 
available from the ioverlander mobile app to train and validate predictions.

## Table of Contents

* [How to use](#How-to-use)
* [Background](#background)
* [Data sources](#data-sources)
* [Data cleaning and aggregation](#Data-cleaning-and-aggregation)
* [EDA and visualization](#Exploratory-Data-Analysis)
* [Modeling](#Modeling)
* [Results](#Results)
* [Conclusions](#Conclusions)
* [Future plans](#future-plans)

## How to use

This project uses jupyter notebooks for the most part, with support functions in `src/helper_funcs.py`.
1. For now, the data has not been uploaded and shared but you can fetch it yourself from iOverlander and get images
using google Static Maps API with these notebooks (located in the notebooks directory): `get_states_zips.ipynb` and `get_images.ipynb`.
2. The training and validation data was examined in `summarize_data.ipynb`
3. The established campground data was examined and obtained in `established_campgrounds.ipynb`
4. The data and models were compiled and developed in `preprocess_and_build_model-binary.ipynb` and run on google colab with `run_sat_models_colab.ipynb`
5. The test data (from states the model had not seen before) was examined in `summarize_test_data.ipynb` to check class balances.
6. You can use `test_model_new_data.ipynb` notebook to load a saved model and test with new data and plot ROC, confusion matrix, example images, and examples of images where the model did not produce the correct classification.
7. The NLP analysis was completed in `NLP_of_descriptions.ipynb`.

## Background

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Data sources

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Data cleaning and aggregation

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Exploratory Data Analysis

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Modeling

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Results

TBA, in the meantime, check the slides: https://paultanger.github.io/CampsitePredict/

## Conclusions

I have been able to successfully use image classification neural networks to detect if a satellite image is a wild camping area or established campground.  In addition, I was able to use Natural Language Processing and K means clustering to split the Wild Camping category into two separate categories.  This could be useful for iOverlander to integrate into the app.

## Future plans

* Clean training data
* Try sobel transformation
* Include additional binary columns such as “bathrooms” etc
* Utilize NLP topics to aid image classification
* Multiclassification
* Train with additional sat images
* Implement class weights
* NLP with higher level tokens than words
* Examine F1 scores
* More dropouts in model