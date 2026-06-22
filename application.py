# the base file
# for UI
from flask import Flask,request,render_template

import numpy as np
import pandas as pd


# pipeline
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline




application=Flask(__name__)

app=application


#route for homepage

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        #post data from the form
        data=CustomData(
              gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
 
 
 
'''The Flask application collects user input through an HTML form. The form data is passed to a CustomData class, which converts the input into a Pandas DataFrame. This DataFrame is sent to the PredictPipeline, where the saved preprocessing pipeline (preprocessor.pkl) is loaded and applied to transform categorical and numerical features into the same format used during training. The transformed data is then passed to the trained machine learning model (model.pkl) for prediction. Finally, the predicted value is returned to Flask and rendered back on the web page for the user.'''