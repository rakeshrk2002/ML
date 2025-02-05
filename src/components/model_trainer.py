import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class model_trainer_Config:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class Model_trainer:
    def __init__(self):
        self.model_config = model_trainer_Config()
        
    def randomized_search(self, model, param_distributions, X_train, y_train, n_iter=20):
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )
        search.fit(X_train, y_train)
        logging.info(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
        return search.best_estimator_
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Catboost Regressor": CatBoostRegressor(verbose=0),
                "XGBoost": XGBRegressor()
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'n_estimators': [50, 100, 200],
                    'min_samples_split': [2, 5, 10]
                },
                "KNN": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                },
                "XGBoost": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9]
                },
                "Catboost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200, 300]
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2]
                },
                "Linear Regression": {}
            }
            
            # Tune models: perform randomized search on those that have parameters defined
            tuned_models = {}
            for model_name, model in models.items():
                if model_name in params and params[model_name]:
                    logging.info(f"Performing hyperparameter tuning for {model_name}")
                    tuned_model = self.randomized_search(model, params[model_name], X_train, y_train, n_iter=30)
                    tuned_models[model_name] = tuned_model
                    
                else:
                    
                    logging.info(f"Fitting {model_name} with default parameters")
                    model.fit(X_train, y_train)
                    tuned_models[model_name] = model
            
            model_report = evaluate_model(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=tuned_models,params=params
            )
            
            if model_report is None:
                raise Custom_Exception("Model evaluation failed: model_report is None", sys)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_individual_model = tuned_models[best_model_name]
            logging.info(f"Best individual model: {best_model_name} with R2 score: {best_model_score}")
            
            # Build a stacking ensemble from all tuned models
            estimators = [(name, model) for name, model in tuned_models.items()]
            stacking_ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression(),
                cv=3,
                n_jobs=-1
            )
            stacking_ensemble.fit(X_train, y_train)
            stacking_score = r2_score(y_test, stacking_ensemble.predict(X_test))
            logging.info(f"Stacking Ensemble R2 score: {stacking_score}")
            
            # Choose the best between the individual best model and the stacking ensemble
            if stacking_score > best_model_score:
                final_model = stacking_ensemble
                final_score = stacking_score
                logging.info("Selected Stacking Ensemble as final model")
            else:
                final_model = best_individual_model
                final_score = best_model_score
                logging.info("Selected Best Individual Model as final model")
            
            # Save the final model
            save_object(
                filepath=self.model_config.trained_model_file_path,
                obj=final_model
            )
            
            predicted = final_model.predict(X_test)
            final_r2 = r2_score(y_test, predicted)
            logging.info(f"Final model R2 score on test set: {final_r2}")
            
            return final_r2 * 100
        
        except Exception as e:
            raise Custom_Exception(e, sys)
