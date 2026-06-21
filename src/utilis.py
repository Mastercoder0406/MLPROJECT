# for the uitlity functions

import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException

import dill # library for creating pickle file 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
    '''The evaluate_models() function is responsible for training, tuning, and evaluating all machine learning models provided in the models dictionary. It receives the training features (X_train), training target values (y_train), testing features (X_test), testing target values (y_test), a dictionary of machine learning models, and a dictionary containing hyperparameter grids for each model. The function begins by creating an empty dictionary called report, which will store the final performance score of each model.

The function then iterates through each model one by one. For every iteration, it extracts the current model object from the models dictionary and retrieves the corresponding hyperparameter grid from the param dictionary. A GridSearchCV object is then created using the model and its parameter grid with cv=3, meaning 3-fold cross-validation will be used. During this process, GridSearchCV automatically tries every possible combination of the specified hyperparameters, trains the model on different folds of the training data, evaluates its performance, and identifies the parameter combination that gives the best result.

Once the optimal hyperparameters are found, the function retrieves them using gs.best_params_ and applies them to the original model using model.set_params(**gs.best_params_). The model is then trained again on the complete training dataset using these best hyperparameter values. After training, predictions are generated for both the training dataset (y_train_pred) and the testing dataset (y_test_pred) using the model's predict() method.

The function then evaluates the model's performance using the R² score metric. The training R² score (train_model_score) indicates how well the model fits the training data, while the testing R² score (test_model_score) measures how accurately the model generalizes to unseen data. Although both scores are calculated, only the testing score is stored in the report dictionary because it is the most reliable indicator of real-world model performance. The model name is used as the key and its testing R² score is used as the value.

After all models have been trained, tuned, and evaluated, the function returns the report dictionary containing the testing R² scores of every model. An example output may look like:

{
    'Random Forest': 0.87,
    'Decision Tree': 0.79,
    'Gradient Boosting': 0.91,
    'Linear Regression': 0.82,
    'XGBRegressor': 0.93,
    'CatBoosting Regressor': 0.92,
    'AdaBoost Regressor': 0.85
}'''


#load object:- loading the pickle file ( pickel file is binary so read in binary)
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)