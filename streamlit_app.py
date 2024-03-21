import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model
model = pickle.load(open('Flask/Templates/model.pkl', 'rb'))

# Define the Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Traffic Volume Prediction")

    # Create input fields for user input
    st.subheader("Enter the features:")
    holiday = st.number_input("Holiday", value=0.0)
    temp = st.number_input("Temperature", value=0.0)
    rain = st.number_input("Rain", value=0.0)
    snow = st.number_input("Snow", value=0.0)
    weather = st.number_input("Weather", value=0.0)
    year = st.number_input("Year", value=0.0)
    month = st.number_input("Month", value=0.0)
    day = st.number_input("Day", value=0.0)
    hours = st.number_input("Hours", value=0.0)
    minutes = st.number_input("Minutes", value=0.0)
    seconds = st.number_input("Seconds", value=0.0)

    # Collect user inputs
    input_data = {
        "holiday": holiday,
        "temp": temp,
        "rain": rain,
        "snow": snow,
        "weather": weather,
        "year": year,
        "month": month,
        "day": day,
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds
    }

    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"Estimated Traffic Volume: {prediction[0]} units")

if __name__ == "__main__":
    main()
