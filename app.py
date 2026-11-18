import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.title("breast cancer prediction (learning project)")

@st.cache_resource
def train_model():
    data = load_breast_cancer(as_frame=True)
    cols = ["mean radius","mean texture","mean perimeter",
            "mean area","mean concavity"]

    x = data.data[cols]
    y = data.target
    model = Pipeline([
        ("scaler",StandardScaler()),
        ("clf",LogisticRegression(max_iter=1000)),
    ])
    model.fit(x,y)
    return model , x , data.target_names
model , x , names = train_model()

inputs = {}
for col in x.columns:
    inputs[col] = st.slider(
        col , float(x[col].min()), float(x[col].max()),float(x[col].mean())
    )

if st.button("predict"):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    st.subheader(f"prediction:{names[pred]}")
    st.write(f"malignant probability: {proba[0]:2f}")

st.caption("learning project only.not for real medical use.")

