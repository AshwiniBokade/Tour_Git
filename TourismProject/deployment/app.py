import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="AshwiniBokade/Tourism-Project-Asgnmt",  # update with your actual repo name
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**.
Please enter the customer details below to get a prediction.
""")

# User Input Form
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (mins)", min_value=1, max_value=60, value=15)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=2)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=30000, step=100)
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'ProductPitched': ProductPitched,
    'Designation': Designation
}])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase Package" if prediction == 1 else "Will Not Purchase Package "
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
