import pandas as pd

car = pd.read_csv("quikr_car.csv")

#print(car.head(10))
#print(car.shape)
#print(car.info())
#print(car["year"].unique())
## Quality
""""
    - year has many non-year values
    - year Object to int 
    
    - Price has Ask for Price
    - Price Object to int 
    
    -Kms_driven has kms with integers
    -Kms_driven Object to int 
    -Kms_driven has nan values
    
    -fuel_type has nan values
    
    -keep first 3 words of name
"""
## Cleaning
backup = car.copy();

car = car[car["year"].str.isnumeric()];
#print(car["year"].astype(int))
#print(car.info())
car["year"] = car["year"].astype(int);
#print(car.info())

car.to_csv("Cleaned_car.csv")

car = car[car["Price"] != 'Ask For Price'];
car['Price'] = car["Price"].str.replace(',','').astype(int);

#print(car.info())

#print(car['kms_driven'])
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
car = car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
#print(car.info())

car = car[~car['fuel_type'].isna()];
#print(car.shape)
car["name"] = car["name"].str.split(" ").str.slice(0,3).str.join(" ");

car = car.reset_index(drop=True)
#print(car)
print(car.info())
car.to_csv("Cleaned_car.csv")