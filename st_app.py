import streamlit as st
import numpy as np
from joblib import load
# import gzip
# import pickle
# from PIL import Image

# image = Image.open('./images/airbnb.png')
# st.image(image, width=100)

# model = pickle.load(gzip.open("rf_reg.pkl"))
# model = load('rf_reg_01_(1).pkl')
model = load('rf_reg_01.pkl')


# Define the Streamlit app
st.title('London Rental Price Prediction')

# Widgets for user input
property_type = st.selectbox('Choose type of the property:', ['Private room', 'Shared room'])
weekends = st.checkbox('Weekends')
person_capacity = st.selectbox('The maximum number of people that can stay in the room:', [2, 3, 4, 5, 6])
host_is_superhost = st.checkbox('Superhost')
multi = st.checkbox('Multi rooms')
bedrooms = st.selectbox('Bedrooms:', [0, 1, 2, 3, 4, 5])
dist = st.slider('Distance from the city centre, miles:', 0.04, 17.0, step=0.1)
metro_dist = st.slider('Metro distance, miles:', 0.01, 9.0, step=0.1)

if property_type == 'Private room':
    room_type_Private = 1
    room_type_Shared = 0
else:
    room_type_Private = 0
    room_type_Shared = 1   

# Unimmutable data
biz = 1
cleanliness_rating = 9.0
guest_satisfaction_overall = 90.0
attr_index = 293
rest_index = 623
haversine_distance = 5727


if property_type == 'Private room':
    room_type_Private = 0
    room_type_Shared = 1
else:
    room_type_Private = 1
    room_type_Shared = 0  


# Button to predict rental price
if st.button('Predict Rental Price'):
    # Prepare the input data for prediction
    input_data = np.array([[
        person_capacity,
        cleanliness_rating,
        guest_satisfaction_overall,
        bedrooms,
        dist,
        metro_dist,
        attr_index,
        rest_index,
        haversine_distance,
        room_type_Private,
        room_type_Shared,
        weekends,
        host_is_superhost,
        multi,
        biz
    ]])

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the predicted rental price
    st.metric(label="Estimated Rental Price:", value=f'£{prediction[0]:.2f}')
    # st.header(f"Estimated Rental Price is £{prediction[0]:.2f}")
