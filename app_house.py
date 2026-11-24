import streamlit as st
import joblib

model = joblib.load("house_model.pkl")

st.title("calforina house price predictor")

MedInc = st.slider("median Income (x $ 10,000)",0.5 , 15.0 , 3.5)
HouseAge = st.slider("house age (years)",1 , 52 , 29)
AveRooms = st.slider("average Rooms",1.0 , 10.0 , 5.4)
AveBedrms = st.slider("average Bedrooms",0.5,5.0,1.0)
Population = st.slider ("Population",3 , 5000 , 1200)
AveOccup = st.slider("average occupancy", 1.0 , 6.0 , 3.0)
Latitude = st.slider("Latitude",32.5,42.0,34.2)
Longitude = st.slider("Longitude",-124.3,-114.3,-118.5)

if st.button("predict"):
    try:
        features = [[float(MedInc),float(HouseAge),float(AveRooms),
                     float(AveBedrms),float(Population),float(AveOccup),
                     float(Latitude),float(Longitude)]]
        st.write(features)
        prediction = model.predict(features)[0]
        price = prediction * 100000
        st.success(f"prediction price: ${price:,.0f}")
    except Exception as e:
        st.error(f"Error:{e}")
