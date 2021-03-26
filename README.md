# Handling imbalanced data and predicting financial fraud with machine learning
This project attempts to use different strategies for handling an imbalanced financial fraud dataset and to build and deploy a machine learning model for predicting fraudulent transactions.

Project blog post can be found here: https://hakon-doksrod.medium.com/financial-fraud-detection-and-handling-imbalanced-data-in-machine-learning-4c9541ae6830



## Table of Contents:
- [Installation](#installation)
- [File descriptions](#file-descriptions)
- [Project Summary](#project-summary)

## Installation
The project should run without issues using Python 3. The project was created using an Anaconda installation. If you are using this, make sure to install the following packages:
- xgboost
- imblearn

The project uses [this synthetic financial fraud dataset from Kaggle](https://www.kaggle.com/ntnu-testimon/paysim1). The file is unfortunately too large to store on GitHub, so to run the code in the notebook you need to download the data from Kaggle and save in the same directory as the notebook.

## File descriptions
- fraud_detection.ipynb - notebook file containing the project code

## Project Summary
The data for this project consists of more than 6 million financial transactions, labeled as either valid or fraudulent transactions. The data is heavily imbalanced, with only 0.13% of the transactions actually being fraudulent.

The goal of the project is to build a machine learning model to predict fraudulent financial transactions. As is expected of financial fraud data, this dataset is heavily imbalanced, and a step along the way to building this model was to find and use different strategies for handling imbalanced data and finding which one yields the best results. I used three strategies - oversampling data, undersampling data and using class weights in the machine learning algorithm itself.

In a real world context, my assumption is that a financial institution will want to first of all minimize the number of false negative predictions (fraudulent transactions flagged as valid) as much as possible, and as a secondary objective to minimize the false positive predictions (valid transactions flagged as fraud) to avoid costly verification work and inconvenienced customers.

Using the XGBoost classifier, i built three models, one for each imbalanced data strategy. All models performed very well in terms of AUC score, all three above 0.99. All three models also performed excellently on false negatives. One model stood out in terms of false positive performance, and that was the model using oversampled data.

As all models were very good on what I defined as the most important metric, namely false negatives, the "winner" was decided by the performance on the secondary metric, false positives. A good performance here can possible save a financial institution a lot of money on unnecessary transaction verification work.
