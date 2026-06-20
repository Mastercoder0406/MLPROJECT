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
                train_array[:,:-1],
                train_array[:,-1],
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
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            
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
            

