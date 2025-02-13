from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extract and validate form data
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),  # Fixed parameter name
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),  # Fixed field name and type
                writing_score=int(request.form.get('writing_score'))   # Fixed field name and type
            )
        except ValueError as e:
            return render_template('home.html', error=f"Invalid input: {str(e)}")

        # Create DataFrame and predict
        pred_df = data.get_data_as_data_frame()
        
        try:
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template(
                'home.html', 
                results=round(results[0], 2),  # Rounded prediction
                input_data=pred_df.iloc[0].to_dict()  # Pass input data for display
            )
        except Exception as e:
            return render_template('home.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)