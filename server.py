import base64
import io
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template


app = Flask(__name__)

def get_model():
   global model
   model = load_model('fall_detector_model.h5')
   
   
   
def preprocess_data(data):
   scaler = MinMaxScaler()
   scaler.fit(data)
   p_data = scaler.transform(data)
   return p_data

print("loading keras model....")
get_model()

@app.route('/', methods=['GET'])
def log():
   return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
   chf = request.form['chf']
   chf = float(chf)
   fst = request.form['chf']
   fst = float(fst)
   pssi = request.form['pssi']
   pssi = float(pssi)
   data = [[chf, fst, pssi]]
   data_p = preprocess_data(data)
   prediction = model.predict(data_p)
   result = float(prediction)
   
   if(result > 0.5):
      return render_template('fall.html')
   elif(result < 0.5):
      return render_template('walk.html')



if __name__ == '__main__':
    
    app.run(debug=True)