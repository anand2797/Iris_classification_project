from flask import Flask, request, render_template
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# create flask app
app = Flask(__name__)

# load the pickle model 
model = joblib.load("logistic_regression_model.pkl")

# load cleaned dataset
data = pd.read_csv('Cleaned_iris_data.csv')

# apply label encoder on target column 
label_encoder = LabelEncoder()
label_encoder.fit(data['Class'])


# homepage of app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    column = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
    input_df = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]], columns=column)
    
    # apply log transformation
    df_log = input_df.apply(lambda x: np.log(x))
    
    # standard Scalling
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)
    
    # prediction
    prediction = model.predict(scaled_input)
    
    
    # inverse label encoder
    predicted_class =label_encoder.inverse_transform(prediction.reshape(-1))[0]
    all_labels = label_encoder.classes_
    return render_template("index.html", predicted_class=predicted_class,
                          sepal_length=sepal_length, sepal_width=sepal_width,petal_length=petal_length,
                          petal_width=petal_width, all_labels=all_labels)

if __name__ == "__main__":
    app.run(debug=True)