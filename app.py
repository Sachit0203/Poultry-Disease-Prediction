from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model 
model = pickle.load(open('C:\Users\hp\final_poultry_project\venv\model.pkl','rb'))
sc = pickle.load(open('C:\Users\hp\final_poultry_project\venv\standscaler.pkl','rb'))
ms = pickle.load(open('C:\Users\hp\final_poultry_project\venv\minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    temperature = request.form['temperature']
    behavioral_score = request.form['behavioral_score']
    respiratory_rate = request.form['respiratory_rate']
    weight = request.form['weight']
    feed_consumption = request.form['feed_consumption']
    age_weeks = request.form['age_weeks']
    

    feature_list = [temperature, behavioral_score, respiratory_rate, weight, feed_consumption, age_weeks]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    disease_dict = {0: "Coccidiosis", 1: "Marek's Disease", 2: "Respiratory Infection"}

    if prediction[0] in disease_dict:
        disease = disease_dict[prediction[0]]
        result = "{} is the disease".format(disease)
    else:
        result = "Sorry, we could not determine the disease"
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)