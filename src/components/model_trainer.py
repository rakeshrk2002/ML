import os
import sys

import pandas as pd 
import numpy as np 
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)

from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class model_trainer_Config:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
    
class Model_trainer:
    def __init__(self):
        self.model_config = model_trainer_Config()
        
    def initiate_model_trainer(self,train_array,test_array):
        
        try:
            
            logging.info("Split training and test input data")
            
            X_train, y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "KNN":KNeighborsRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBoost":XGBRegressor()
            }
            
            model_report:dict = evaluate_model(X_train=X_train,
                                               y_train = y_train,
                                               X_test=X_test,
                                               y_test=y_test,
                                               models=models)
            
            if model_report is None:
                raise Custom_Exception("Model evaluation failed: model_report is None", sys)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            
            # best_model_score = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            
            
            if best_model_score < 0.6:
                raise Custom_Exception("No best Model Found")
            logging.info(f"Best Model found in both training and testing dataset")
            
            save_object(
                filepath=self.model_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test,predicted)
            
            return r2_square*100
        
        except Exception as e:
            
            raise Custom_Exception(e,sys)