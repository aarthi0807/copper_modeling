# Copper Modeling
With the given dataset, we can predict selling price with Regression model and status of the item with classification model.<br/>
Given data is cleaned, tranformed with log function appropriately.<br/>
Null value analysis has been made and unnecessary datapoints and features have been removed.<br/>
After cleaning the data, we can see normal distribution of the features.<br/>
## Data preprocessing
StandardScaler have been applied on the numerical features.<br/>
OneHotEncoder,OrdinalEncoder is applied on the categorical features appropriately.<br/>
After preprocessing data, will be in numpy array and everything is concatenated and splitted into Train and Test data.<br/>
## Model Building
With X train and Y train,Tree based Regression and Classification model is built with best params which was found from GridSeachCV.<br/>
## Evaluation
Metrics have been calculated for train and test data for both the models and we could see the scores are comparable.

# Streamlit App: https://coppermodeling-xtxry4sqytqtghkmhfyh7j.streamlit.app/
This app is designed to enter the feature values in UI and the prediction is made with the best model we have built.
