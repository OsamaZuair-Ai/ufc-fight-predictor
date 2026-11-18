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

# Load Trained Model, Scaler, Imputer, Features List, and Fighter Dataset
@st.cache_resource
def load_assets():
    model = joblib.load('ufc_rf_model.pkl')
    scaler = joblib.load('ufc_scaler.pkl')
    imputer = joblib.load('ufc_imputer.pkl')
    model_features = joblib.load('model_features.pkl')
    df_fighters = pd.read_csv('Fighters.csv')
    return model, scaler, imputer, model_features, df_fighters

# Execute asset loading properly
try:
    model, scaler, imputer, model_features, df_fighters = load_assets()
    st.sidebar.success("Model & Assets loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading files: {e}")
    st.stop()

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
                return np.nan

        # Extract stats for both fighters
        w1, l1 = clean_val(f1_data.get('W', 0)), clean_val(f1_data.get('L', 0))
        w2, l2 = clean_val(f2_data.get('W', 0)), clean_val(f2_data.get('L', 0))

        win_rate_1 = w1 / (w1 + l1 + 1e-5)
        win_rate_2 = w2 / (w2 + l2 + 1e-5)

        ht_1, ht_2 = clean_val(f1_data.get('Ht.')), clean_val(f2_data.get('Ht.'))
        wt_1, wt_2 = clean_val(f1_data.get('Wt.')), clean_val(f2_data.get('Wt.'))
        reach_1, reach_2 = clean_val(f1_data.get('Reach')), clean_val(f2_data.get('Reach'))

        diff_win_rate = win_rate_1 - win_rate_2
        diff_reach = reach_1 - reach_2 if (not np.isnan(reach_1) and not np.isnan(reach_2)) else np.nan
        diff_ht = ht_1 - ht_2 if (not np.isnan(ht_1) and not np.isnan(ht_2)) else np.nan
        diff_total_fights = (w1 + l1) - (w2 + l2)

        # Build raw dict matching original feature names
        input_data = {
            'Ht._1': ht_1, 'Wt._1': wt_1, 'Reach_1': reach_1,
            'Ht._2': ht_2, 'Wt._2': wt_2, 'Reach_2': reach_2,
            'Diff_Win_Rate': diff_win_rate,
            'Diff_Reach': diff_reach,
            'Diff_Ht': diff_ht,
            'Diff_Total_Fights': diff_total_fights,
            'Weight_Class': f1_data.get('Weight_Class', 'Lightweight'),
            'Stance_1': f1_data.get('Stance', 'Orthodox'),
            'Stance_2': f2_data.get('Stance', 'Orthodox')
        }

        # Convert input to DataFrame & apply One-Hot Encoding
        input_df = pd.DataFrame([input_data])
        categorical_cols = ['Weight_Class', 'Stance_1', 'Stance_2']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, dtype=int)

        # Reindex columns to strictly match trained model features
        final_input = input_encoded.reindex(columns=model_features, fill_value=0)

        # Impute missing values & scale features
        features_clean = imputer.transform(final_input)
        scaled_features = scaler.transform(features_clean)

        # Predict outcome using model
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