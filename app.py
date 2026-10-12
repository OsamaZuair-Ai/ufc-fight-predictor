import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered"
)

# App Title and Description
st.title("🥊 UFC Fight Winner Predictor")
st.markdown("Select two fighters from the database to predict the match outcome using Machine Learning.")

# Load Trained Model, Scaler, Imputer, and Fighter Dataset
@st.cache_resource
def load_assets():
    model = joblib.load('ufc_rf_model.pkl')
    scaler = joblib.load('ufc_scaler.pkl')
    imputer = joblib.load('ufc_imputer.pkl')
    df_fighters = pd.read_csv('Fighters.csv')
    return model, scaler, imputer, df_fighters

try:
    model, scaler, imputer, df_fighters = load_assets()
    st.sidebar.success("Random Forest Model & Assets loaded successfully!")
except Exception as e:
    st.sidebar.error("Error loading files. Ensure 'Fighters.csv', 'ufc_rf_model.pkl', 'ufc_scaler.pkl', and 'ufc_imputer.pkl' exist in directory.")

# Extract sorted unique list of fighter names
fighter_names = sorted(df_fighters['Full Name'].dropna().unique().tolist())

# Interface Columns for Fighter Selection
col1, col2 = st.columns(2)

with col1:
    st.header("🔴 Red Corner")
    fighter_1 = st.selectbox("Select Fighter 1", fighter_names, index=0)

with col2:
    st.header("🔵 Blue Corner")
    fighter_2 = st.selectbox("Select Fighter 2", fighter_names, index=1 if len(fighter_names) > 1 else 0)

# Display Stats Comparison Card
if fighter_1 and fighter_2:
    f1_data = df_fighters[df_fighters['Full Name'] == fighter_1].iloc[0]
    f2_data = df_fighters[df_fighters['Full Name'] == fighter_2].iloc[0]
    
    st.markdown("---")
    st.subheader("📊 Fighter Comparison")
    
    stat_col1, stat_col2 = st.columns(2)
    
    with stat_col1:
        st.info(f"**{fighter_1}**\n\n"
                f"• Record: {f1_data.get('W', 0)}W - {f1_data.get('L', 0)}L\n"
                f"• Height: {f1_data.get('Ht.', 'N/A')}\n"
                f"• Weight: {f1_data.get('Wt.', 'N/A')} lbs\n"
                f"• Reach: {f1_data.get('Reach', 'N/A')}\n"
                f"• Stance: {f1_data.get('Stance', 'N/A')}")
        
    with stat_col2:
        st.info(f"**{fighter_2}**\n\n"
                f"• Record: {f2_data.get('W', 0)}W - {f2_data.get('L', 0)}L\n"
                f"• Height: {f2_data.get('Ht.', 'N/A')}\n"
                f"• Weight: {f2_data.get('Wt.', 'N/A')} lbs\n"
                f"• Reach: {f2_data.get('Reach', 'N/A')}\n"
                f"• Stance: {f2_data.get('Stance', 'N/A')}")

# Prediction Logic Execution
st.markdown("---")
if st.button("🚀 Predict Winner", use_container_width=True):
    if fighter_1 == fighter_2:
        st.warning("⚠️ Please select two different fighters!")
    else:
        def clean_val(val):
            try:
                return float(val)
            except:
                return 0.0

        # Strict Female Identification Check using Names & Weight/Category Attributes
        known_female_keywords = [
            'rose namajunas', 'weili zhang', 'amanda nunes', 'valentina shevchenko', 
            'holly holm', 'jessica andrade', 'marina rodriguez', 'yan xiaonan', 
            'joanna jedrzejczyk', 'carla esparza', 'julianna pena', 'alexa grasso',
            'erin blanchfield', 'manon fiorot', 'tatiana suarez', 'kayla harrison'
        ]
        
        def check_if_female(fighter_name, fighter_row):
            name_lower = str(fighter_name).lower()
            if any(fem in name_lower for fem in known_female_keywords):
                return True
            
            # Check weight class string if available
            weight_class_val = str(fighter_row.get('Weight_Class', '')).lower()
            if 'women' in weight_class_val or 'female' in weight_class_val:
                return True
                
            return False

        f1_is_female = check_if_female(fighter_1, f1_data)
        f2_is_female = check_if_female(fighter_2, f2_data)

        if f1_is_female != f2_is_female:
            st.error("🚫 Invalid Matchup: Intergender fights (Male vs Female) are strictly prohibited. Please select fighters from the same division.")
        else:
            # Calculate Raw Statistics & Rates
            w1, l1 = clean_val(f1_data.get('W', 0)), clean_val(f1_data.get('L', 0))
            w2, l2 = clean_val(f2_data.get('W', 0)), clean_val(f2_data.get('L', 0))

            win_rate_1 = w1 / (w1 + l1 + 1e-5)
            win_rate_2 = w2 / (w2 + l2 + 1e-5)

            diff_win_rate = win_rate_1 - win_rate_2
            diff_reach = clean_val(f1_data.get('Reach', 0)) - clean_val(f2_data.get('Reach', 0))
            diff_ht = clean_val(f1_data.get('Ht.', 0)) - clean_val(f2_data.get('Ht.', 0))
            diff_total_fights = (w1 + l1) - (w2 + l2)

            # Align features array to expected shape
            expected_features_count = scaler.n_features_in_
            features_array = np.zeros((1, expected_features_count))

            # Insert engineered features into the leading indices
            features_array[0, :4] = [diff_win_rate, diff_reach, diff_ht, diff_total_fights]

            # Impute and scale input features
            features_clean = imputer.transform(features_array)
            scaled_features = scaler.transform(features_clean)

            # Generate Prediction using Random Forest Model
            prediction = model.predict(scaled_features)[0]
            prediction_proba = model.predict_proba(scaled_features)[0]

            # Display Winner Result
            st.markdown("### 🏆 Predicted Winner")
            if prediction == 1:
                winner = fighter_1
                confidence = prediction_proba[1] * 100
            else:
                winner = fighter_2
                confidence = prediction_proba[0] * 100
                
            st.success(f"🎉 **{winner}** is predicted to WIN! (Confidence: {confidence:.1f}%)")

       