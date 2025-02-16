from flask import Flask, request, render_template
import pandas as pd
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# artifacts_dir = "artifacts"
# print("Current working directory:", os.getcwd(artifacts_dir))

# print("Artifacts directory absolute path:", os.path.abspath(artifacts_dir))
# print("Artifacts directory exists:", os.path.exists(artifacts_dir))
# print("Artifacts directory is writable:", os.access(artifacts_dir, os.W_OK))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        
        return render_template('home.html')
    else:
        try:
            
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race/ethnicity'),  
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'), 
                writing_score=request.form.get('writing_score')
            )
            print("Form Data:", request.form)  
        except ValueError as e:
            print(f"ValueError: {e}") 
            return render_template('home.html', error=f"Invalid input: {str(e)}")

        pred_df = data.get_data_as_dataframe()
        print("DataFrame:", pred_df) 

        try:
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("Prediction Results:", results)
            # return render_template('home.html', results=results)

            return render_template('home.html', results=round(results[0], 2), input_data=pred_df.iloc[0].to_dict())
        except Exception as e:
            print(f"Prediction Error: {e}")
            return render_template('home.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', debug=True)
    