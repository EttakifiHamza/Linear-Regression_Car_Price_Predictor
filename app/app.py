from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
app =Flask(__name__);
model = pickle.load(open("../dataset/LinearRegresionModeCar.pkl","rb"))
car =  pd.read_csv("../dataset/Cleaned_car.csv");
@app.route("/")
@app.route("/index")
def index():
    companies = sorted(car["company"].unique())
    car_model = sorted(car["name"].unique())
    year = sorted(car["year"].unique(),reverse=True)
    fuel_type = sorted(car["fuel_type"].unique())
    return render_template("index.html",companies=companies,car_model=car_model,year=year,fuel_type=fuel_type)
@app.route("/predict",methods=["POST","GET"])
def predict():
    company = request.form.get("companies")
    car_model = request.form.get("car_model")
    year = int(request.form.get("year"))
    fuel_type = request.form.get("fuel_type")
    kilo_driven =int(request.form.get("kilo_driven"));
    columns = ['Unnamed: 0', 'name', 'company', 'year', 'kms_driven', 'fuel_type']
    prediction = model.predict(pd.DataFrame(columns=columns,data=np.array([0,car_model,company,year,kilo_driven,fuel_type]).reshape(1,6)))

    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)
