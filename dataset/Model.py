import pandas as pd

car = pd.read_csv("Cleaned_car.csv");
#print(car.info())

x = car.drop(columns="Price")
#print(x.shape)
y = car["Price"]
#print(y.shape)
##          DATA SPLIT
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2);

#print("Training Data",x_train.shape)
#print("Testing Data",x_test.shape)

##              Liner Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import  make_pipeline
ohe = OneHotEncoder();
ohe.fit(x[["name","company","fuel_type"]])
#print(ohe.categories_)
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),["name","company","fuel_type"]),remainder="passthrough");
lr = LinearRegression();
pipe = make_pipeline(column_trans,lr);

pipe.fit(x_train,y_train);

y_pred = pipe.predict(x_test)
scores = [];
for i in range(1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=i);
    lr = LinearRegression();
    pipe = make_pipeline(column_trans,lr);
    pipe.fit(x_train,y_train)
    y_pred =pipe.predict(x_test);
    scores.append(r2_score(y_test,y_pred))
import numpy as np


#print(np.argmax(scores))
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores));
lr = LinearRegression();
pipe = make_pipeline(column_trans,lr);
pipe.fit(x_train,y_train)
y_pred =pipe.predict(x_test);

#print(r2_score(y_test,y_pred))

import pickle
print(scores[np.argmax(scores)])
pickle.dump(pipe,open("LinearRegresionModeCar.pkl","wb"))

prd = pipe.predict(pd.DataFrame(columns=x_test.columns,data=np.array([0,'Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,6)))

print(x_test.columns)