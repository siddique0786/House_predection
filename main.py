import pandas as pd
from flask import Flask ,render_template , request
import numpy as np
import pickle

app = Flask(__name__,template_folder='template')
data=pd.read_csv("Cleaned_data.csv")
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():

    locations=sorted(data['location'].unique())
    locations.insert(0,"Select Locations")
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    locations=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=int(request.form.get('bath'))
    sqft=int(request.form.get('total_sqft'))


    #print(locations,bhk,bath,sqft)
    input = pd.DataFrame([[locations,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0] * 1e5


    return str(np.round(prediction,2))



if __name__=="__main__":
    app.run(debug=True,port=5001)