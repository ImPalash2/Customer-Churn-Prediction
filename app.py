import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
# print(tf.__version__)

## load the trained model
model = tf.keras.models.load_model('./model.h5')

## Load the encoders and scalers
with open('./onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('./label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('./scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Start using streamlit app
st.title("Customer Churn prediction")

# user input
# print(onehot_encoder_geo.categories_[0])
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', min_value=18, max_value=92)
balance = st.number_input('Balance', min_value=0)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0,1])
is_active_member = st.selectbox('Has active member', [0,1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary],
    # 'Geography' : geography,
})

# onehot encode 'geography'
encoded_geo = onehot_encoder_geo.transform([[geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## concat one-hot encoded data with input data
input_data = pd.concat([input_data.reset_index(drop=True), encoded_geo_df], axis=1)

## scale the input data
scaled_input_data = scaler.transform(input_data)

## predict churn
prediction = model.predict(scaled_input_data)
prediction_probablity = prediction[0][0]

st.write(f'Churn Probablity: {prediction_probablity:.2f}')
if(prediction_probablity > 0.5):
    st.write('The Customer is likely to churn.')
else:
    st.write('The Customer is not likely to churn.')
