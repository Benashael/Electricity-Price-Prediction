# Importing packages for deployment
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
from streamlit_option_menu import option_menu

# Importing packages for model building 
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from category_encoders import OneHotEncoder, OrdinalEncoder
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("C:/Users/Arul Selvaraj/Desktop/EPP_Intern_1/energy_cleaned_dataset.csv")

# Filling null values
data.fillna(method='ffill', inplace=True)

# Split the data into features and target
X = data.drop(['price_actual','time'], axis=1)
y = data['price_actual']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

# Ordinal Encoder to transform Seasons column
ordinal = OrdinalEncoder()
ordinal_fit = ordinal.fit(X_train)
XT_train = ordinal.transform(X_train)
XT_val = ordinal.transform(X_val)

# Simple imputer to fill nan values, then transform sets
simp = SimpleImputer(strategy='mean')
simp_fit = simp.fit(XT_train)
XT_train = simp.transform(XT_train)
XT_val = simp.transform(XT_val)


# Model selection
model_names = ['Linear Regression', 'Ridge Regression', 'Random Forest Regressor', 'XGBoost Regressor']
model_choice = st.selectbox('Choose a model for prediction:', model_names)

if model_choice == 'Linear Regression':
    # Train the linear regression model
    model=LinearRegression()
    model.fit(XT_train,y_train)
elif model_choice == 'Ridge Regression':
    # Train the Ridge regression model
    model=Ridge()
    model.fit(XT_train,y_train)
elif model_choice == 'Random Forest Regressor':
    # Train the random forest regressor model
    model=RandomForestRegressor()        
    model.fit(XT_train,y_train)
elif model_choice == 'XGBoost Regressor':
    # Train the XGBoost regressor model
    model=XGBRegressor()
    model.fit(XT_train,y_train)

# Create the web application
st.title('Electricity Price Prediction')
st.write('Enter the values below to predict the electricity price:')

# Input form for user input
generation_biomass=st.number_input('Generation Biomass')
generation_fossil_brown_coal=st.number_input('Generation Fossil (Coal)')
generation_fossil_gas=st.number_input('Generation Fossil (Gas)')
generation_fossil_hard_coal=st.number_input('Generation Fossil (Hard Coal)')
generation_fossil_oil=st.number_input('Generation Fossil (Oil)')
generation_hydro_pumped_storage_consumption=st.number_input('Generation Hydro Pumped')
generation_hydro_run_of_river_and_poundage=st.number_input('Generation River')
generation_hydro_water_reservoir=st.number_input('Generation Reservior')
generation_nuclear=st.number_input('Generation Nuclear')
generation_other=st.number_input('Generation Other')
generation_other_renewable=st.number_input('Generation Other (Renewable)')
generation_solar=st.number_input('Generation Solar')             
generation_waste=st.number_input('Generation Waste')
generation_wind_onshore=st.number_input('Generation Wind')
total_load_actual=st.number_input('Total Load')
seasons_list=['automn', 'spring', 'summer', 'winter']
season_categorical=st.selectbox('Choose a season:', seasons_list)
seasons = {
    'summer': 1,
    'winter': 2,
    'spring': 3
}
if season_categorical in seasons:
    season = seasons[season_categorical]
    	
# Prediction              
prediction = model.predict([[generation_biomass, generation_fossil_brown_coal, generation_fossil_gas,generation_fossil_hard_coal,generation_fossil_oil,generation_hydro_pumped_storage_consumption,generation_hydro_run_of_river_and_poundage,generation_hydro_water_reservoir,generation_nuclear,generation_other,generation_other_renewable,generation_solar,generation_waste,generation_wind_onshore,total_load_actual,season]])
st.write('Predicted Electricity Price:', prediction[0])

# To display dataset 
st.subheader('Dataset')
st.write(data)






