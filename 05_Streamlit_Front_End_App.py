import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Ford Car Price Predictor",page_icon="🚘",layout="centered")

# Load model & columns
@st.cache_resource
def load_artifacts():
    model = joblib.load("06_Best_Model.pkl")
    columns = joblib.load("07_Columns.pkl")
    return model, columns

model, columns = load_artifacts()

# Session State for steps
if "step" not in st.session_state:
    st.session_state.step = 1

# Header
st.title("🚘 Ford Car Price Predictor")
st.markdown("Developed by Moiz Imam | ML Engineer")
st.divider()

# Progress Bar
st.progress(st.session_state.step / 3,text=f"Step {st.session_state.step} of 3")

# Step 1
if st.session_state.step == 1:
    st.subheader("Step 1 — Car Info")

    st.session_state.model_name = st.selectbox("Car Model",[
        'Fiesta', 'Focus', 'Puma', 'Kuga', 'EcoSport', 'C-MAX', 'Mondeo',
        'Ka+', 'Tourneo Custom', 'S-MAX', 'B-MAX', 'Edge', 'Tourneo Connect',
        'Grand C-MAX', 'KA', 'Galaxy', 'Mustang', 'Grand Tourneo Connect',
        'Fusion', 'Ranger', 'Streetka', 'Escort', 'Transit Tourneo'
    ],index = 0)
    st.session_state.year = st.number_input("Year",min_value=1990,max_value=2025)
    st.session_state.transmission = st.selectbox("Transmission",["Manual","Automatic","Semi-Auto"])
    st.session_state.fuel_type = st.selectbox("Fuel Type",["Petrol","Diesel","Hybrid","Eletric","Other"])
    st.session_state.engine_size = st.number_input("Engine Size (L)",min_value=0.0,max_value=5.0,value=2.0)

    if st.button("Next →",use_container_width=True):
        st.session_state.step = 2
        st.rerun()
# Step 2
elif st.session_state.step == 2:
    st.subheader("Step 2 — Usage Info")        

    st.session_state.mileage = st.number_input("Mileage (Miles)",min_value=0,max_value=300000,value=20000,step=1000)
    st.session_state.mpg = st.number_input("MPG (Fuel Efficiency)",min_value=10.0,max_value=150.0, value=55.0)
    st.session_state.tax = st.number_input("Road Tax (£)",min_value=0, max_value=600, value=145)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back",use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next →",use_container_width=True):
            st.session_state.step = 3
            st.rerun()
# Step 3
elif st.session_state.step == 3:
    st.subheader("Step 3 — Prediction")
    if st.button("← Back",use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    if st.button("🔍 Predict Price",use_container_width=True):
        # Feature engineering
        car_age = 2026 - st.session_state.year
        mileage_per_year = st.session_state.mileage / (car_age + 1)
        is_new = 1 if car_age <= 1 else 0            

        # Build input dict
        input_dict = {col: 0 for col in columns}

        # Numerical
        input_dict['year'] = st.session_state.year
        input_dict['mileage'] = st.session_state.mileage
        input_dict['tax'] = st.session_state.tax
        input_dict['mpg'] = st.session_state.mpg
        input_dict['engineSize'] = st.session_state.engine_size
        input_dict['car_age'] = car_age
        input_dict['mileage_per_year'] = mileage_per_year
        input_dict['is_new'] = is_new

        # One-hot
        model_col = f'model_{st.session_state.model_name}'
        if model_col in input_dict:
            input_dict[model_col] = 1

        if st.session_state.transmission == 'Manual':
            input_dict['transmission_Manual'] = 1
        elif st.session_state.transmission == 'Semi-Auto':
            input_dict['transmission_Semi-Auto'] = 1

        if st.session_state.fuel_type == 'Electric':
            input_dict['fuelType_Electric'] = 1
        elif st.session_state.fuel_type == 'Hybrid':
            input_dict['fuelType_Hybrid'] = 1
        elif st.session_state.fuel_type == 'Other':
            input_dict['fuelType_Other'] = 1
        elif st.session_state.fuel_type == 'Petrol':
            input_dict['fuelType_Petrol'] = 1
        # Predict
        input_df = pd.DataFrame([input_dict])
        predicted_price = model.predict(input_df)[0]
        # Convert to USD
        predicted_usd = predicted_price * 1.27

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"### 💷 Price in GBP\n# £{predicted_price:,.0f}")
        with col2:
            st.success(f"### 💵 Price in USD\n# ${predicted_usd:,.0f}")

        st.balloons()

        st.divider()
        # Summary
        st.markdown("**Your Input Summary:**")
        st.write(f"- Model: {st.session_state.model_name}")
        st.write(f"- Year: {st.session_state.year}")
        st.write(f"- Transmission: {st.session_state.transmission}")
        st.write(f"- Fuel Type: {st.session_state.fuel_type}")
        st.write(f"- Mileage: {st.session_state.mileage:,} miles")
        st.write(f"- MPG: {st.session_state.mpg}")
        st.write(f"- Tax: £{st.session_state.tax}")
        st.write(f"- Engine Size: {st.session_state.engine_size}L")

        if st.button("🔄 Start Over",use_container_width=True):
            st.session_state.step == 1
            st.rerun()    
