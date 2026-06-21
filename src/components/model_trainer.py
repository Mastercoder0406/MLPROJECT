# code related to model training and related fucntions and operations


import os
import sys
from dataclasses import dataclass


#model libraries
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.tree import  DecisionTreeRegressor
from xgboost import XGBRegressor



#logging and exception
from src.logger import logging
from src.exception import CustomException

from src.utilis import save_object,evaluate_models





@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test infor")
            X_train,y_train, X_test,y_test=(
                #train test split data
                train_array[:,:-1], # take all the coloums except the result
                train_array[:,-1],# take only result coloum
                test_array[:,:-1],
                test_array[:,-1])
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            #parameter
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ##to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            
            ##get best modle name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException(" No best Model found")
            logging.info("No best model on both Training and testing data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            Model_r2_score=r2_score(y_test,predicted)
            return Model_r2_score
            
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
'''The ModelTrainer component is responsible for selecting, training, evaluating, and saving the best machine learning model for the student performance prediction project. When the initiate_model_trainer() method is called, it receives the preprocessed training and testing datasets as NumPy arrays. The method first separates the input features and target variable from both datasets by assigning all columns except the last column to X_train and X_test, and the last column (the target variable) to y_train and y_test. After preparing the data, it creates a collection of regression algorithms including Random Forest Regressor, Decision Tree Regressor, Gradient Boosting Regressor, Linear Regression, XGBoost Regressor, CatBoost Regressor, and AdaBoost Regressor. Along with these models, a hyperparameter dictionary is defined for each algorithm, specifying different parameter combinations that will be explored during tuning.

The method then calls the evaluate_models() function, passing the datasets, models, and hyperparameter grids. This function performs hyperparameter tuning using GridSearchCV, trains every model with its best parameter combination, evaluates its performance using the R² score, and returns a report containing the testing scores of all models. The ModelTrainer then analyzes this report to identify the model with the highest testing R² score. Once the best-performing model is found, its score is checked against a minimum threshold of 0.6 to ensure acceptable performance. If no model achieves this threshold, an exception is raised indicating that no suitable model was found.

If a valid model is identified, the model object is retrieved from the model dictionary and saved to the artifacts/model.pkl file using the save_object() utility function. Saving the model allows it to be reused later for prediction without retraining. After saving, the best model is used to generate predictions on the test dataset, and a final R² score is calculated between the actual target values and the predicted values. This score represents the performance of the selected model on unseen data and is returned by the method. Overall, the ModelTrainer class automates the complete model selection pipeline, including dataset preparation, hyperparameter tuning, model comparison, best model selection, model persistence, and final performance evaluation.'''

