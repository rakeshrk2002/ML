import sys
import pandas as pd 

from src.exception import Custom_Exception
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        
        try:
            model_path = os.path.join(r'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(r'artifacts', 'preprocessor.pkl')

            print(f"Model path: {os.path.abspath(model_path)}")
            print(f"Preprocessor path: {os.path.abspath(preprocessor_path)}")
            print(f"Model exists: {os.path.exists(model_path)}")
            print(f"Preprocessor exists: {os.path.exists(preprocessor_path)}")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("File exists:", os.path.exists(model_path))
            print("File size (bytes):", os.path.getsize(model_path))
        
            data_scaled = preprocessor.transform(features)
            predicted_data = model.predict(data_scaled)
            
            return predicted_data
        
        except Exception as e:
            raise Custom_Exception(e,sys)

class CustomData:
    def __init__(self,gender:str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):
        
        try:
            self.gender = gender
            self.race_ethnicity = race_ethnicity
            self.parental_level_of_education = parental_level_of_education
            self.lunch = lunch
            self.test_preparation_course = test_preparation_course
            self.reading_score = reading_score
            self.writing_score = writing_score
        
        except Exception as e:
            raise Custom_Exception(e,sys)
        
    def get_data_as_data_frame(self):
        try:
            custome_data_input_dict = {
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score],
            }
                
            return pd.DataFrame(custome_data_input_dict)
            
        except Exception as e:
            raise Custom_Exception(e,sys)
        