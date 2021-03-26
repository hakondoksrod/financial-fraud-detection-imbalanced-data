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

## File descriptions
- Starbucks.ipynb - notebook file containing the project code
- test_results.py - script to display results of optimization strategy
- Test.csv - file with the test portion of the dataset
- training.csv - file with the training portion of the dataset

## Project Summary
The data for this project consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion.

The goal is to maximize the following metrics:
- **Incremental Response Rate(IRR)**
    - IRR shows how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion
- **Net Incremental Revenue (NIR)**
    - NIR shows the net revenue of the promotion after deducting the cost of distributing the promotion

After calculating the metrics and performing hypothesis tests to verify the significance of the results, we could see that the promotion leads to an increase in IRR, meaning more people purchased the product with the promotion, but a negative NIR, meaning that Starbucks would lose money distributing the promotion to all customers, due to the cost of distribution.

Using a machine learning classification model, I could optimize the promotion strategy by identifying which customers are likely to purchase the product upon receiving the promotion, enabling Starbucks to do targeted promotion. 

Using a Random Forest Classifier yielded a positive result, but lower than the baseline model which has an IRR of 0.0188 and NIR of 189.45. Using an XGBoost classifier, however, improved the model and produced an IRR of 0.0214 and NIR of 313.25, which is an improvement on the baseline model.
