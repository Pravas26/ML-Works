import streamlit as st
import pandas as pd
import pickle

# Load cleaned dataset

df = pd.read_csv("Hyderbad_House_price.csv")

# Extract BHK options from title
bhk_options = sorted(df['title'].str.extract(r'(\d+)').dropna()[0].astype(int).unique())

# Extract unique locations
locations = sorted(df['location'].dropna().unique())

# Load trained model

with open("model/linear_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI

st.title("House Price Predictor")
st.subheader("Location: Hyderabad")

area = st.number_input(
    "Total Area (in sqft)",
    min_value=300,
    max_value=10000,
    value=1000,
    step=50
)

bhk = st.selectbox("BHK", bhk_options)

location = st.selectbox("Location", locations)

if st.button("Predict Price"):
    # Prepare input dataframe
    input_df = pd.DataFrame([{
        'area_insqft': area,
        'location': location,
        'BHK': bhk
    }])

    # Make prediction
    price = model.predict(input_df)[0]

    # Clamp negative to zero
    price = max(price, 0)

    st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")
