# Insurance-Claims-Prediction-CDSAML
This project was a part of my summer internship at Centre for Data Science and Applied Machine Learning at PES University.
# Problem Statement
The aim of the project was to come up with a machine learning based model for prediction of insurance claims given the historical data.
# Approach
The problem was theoretically a regression problem by a classification approach was applied to it to predict the bracket of insurance claim the given insurance case will fall in. The model was made only for automobile insurance claims.
# Data Cleaning
The given data was cleaned by removing the outliers and filling up the missing values for various features.
# Data Preprocessing
The categorical features were encoded and the numeric features were normalized.
# Feature Selection
Feature selection was applied on the data to select which features will be used for making the prediction model.
# Clustering
The given data was clustered based on incurred amount to find some similarity between the claims cases. The bins for the claim prediction were formed accordingly. This clustering was a self made heirarchial clustering model.
# Prediction Model
The predictive model was a neural network based classification model which was able to classify various cases into the claim bins according to the features given as input.
# Outcome
The model was able to predict the claim bin for a given claim with an accuracy of 51.28%
# Scope For Improvement
The accuracy of the prediction needs to be improved.
Instead of prediting range, the model should be extended to predict the exact claim amount value.
