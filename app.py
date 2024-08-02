import pickle
from flask import Flask,request,app,render_template,jsonify,url_for
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the model
model=pickle.load(open('./model.pkl','rb'))
scaler=pickle.load(open('./scalling.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.json['data']
    print(data)
    
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction=model.predict(new_data)
    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(port=5000,debug=True)
    