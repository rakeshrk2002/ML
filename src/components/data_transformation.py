import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object


# class for creating a pickle file in artifacts folder
@dataclass
class DataTransformation_config:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
    
#  Data Transformation class to handle data transformation process
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformation_config()
        
    def get_data_transformer(self):
        
        try:
            num_features = [
                'writing score', 'reading score'
            ]
            
            cat_features = [
                'gender', 'race/ethnicity', 'parental level of education',
                'lunch', 'test preparation course'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]
            )
            
            logging.info(f"Numerical columns Standard scaler is completed : {num_features}")
            logging.info(f"Categorical columns OneHot encoding is completed : {cat_features}")
            
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline',num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline, cat_features)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise Custom_Exception(e,sys)
            
    # A method which is used to read the training and testing datasets from the specified paths
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data")
            
            logging.info("Obtaining preprocessor Object")
            
            preprocessing_obj = self.get_data_transformer()
            
            target_column = "math score"
            numerical_columns = ['writing score','reading score']
            
            # Splitting features and targets
            input_feature_train_df = train_df.drop(columns=[target_column],axis = 1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns = [target_column],axis = 1)
            target_feature_test_df = test_df[target_column]
            
            logging.info("Applying preprocessing object on training and testing dataset")
            
            # This one applies the preprocessing pipeline to training and testing datasets
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            # This combines the preprocessed features and target variable into a single array for training and testing
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved the preprocessing object")
            
            save_object(
                
                filepath = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
                
            )
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            
            raise Custom_Exception(e,sys)
