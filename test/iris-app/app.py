# streamlit-only: run with `streamlit run app.py`, not `python app.py`
# app.py
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# --- load the saved pipeline ONCE, not on every interaction
@st.cache_resource
def load_model():
    return joblib.load("models/iris_model.joblib")

bundle = load_model()
pipeline = bundle["pipeline"]

st.title("🌸 Iris Species Classifier")
st.write("Enter four measurements in centimetres and the model will name the species.")

# --- collect the inputs
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 1.3, 0.1)

# --- build a one-row DataFrame with the SAME column names as training
sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                      columns=bundle["features"])

# --- predict
if st.button("Predict species", type="primary"):
    prediction = pipeline.predict(sample)[0]
    probabilities = pipeline.predict_proba(sample)[0]

    st.success(f"Predicted species: **{bundle['classes'][prediction]}**")

    st.subheader("How confident is it?")
    st.bar_chart(pd.Series(probabilities, index=bundle["classes"]))

    confidence = probabilities.max()
    if confidence < 0.60:
        st.warning(f"Low confidence ({confidence:.0%}). Treat this as a guess.")

# --- honesty, on screen
st.caption(f"Model test accuracy: {bundle['accuracy']:.1%}. "
           "Trained on 150 flowers — a teaching dataset, not a botanical tool.")