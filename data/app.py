import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('ipl_model_winner.pkl')

st.title("🏏 IPL Winner Prediction App")

# Dropdown values
teams = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Rajasthan Royals",
    "Punjab Kings", "Delhi Capitals", "Gujarat Titans", "Lucknow Super Giants"
]

venues = [
    "Wankhede Stadium", "Eden Gardens", "Arun Jaitley Stadium",
    "M Chinnaswamy Stadium", "Narendra Modi Stadium",
    "Maharashtra Cricket Association Stadium", "Sawai Mansingh Stadium"
]

toss_decisions = ["bat", "field"]

# User inputs
team1 = st.selectbox("Select Team 1", teams)
team2 = st.selectbox("Select Team 2", teams)
venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Who won the toss?", teams)
toss_decision = st.selectbox("Toss decision", toss_decisions)

# Predict Button
if st.button("Predict Winner"):
    df_input = pd.DataFrame([[team1, team2, toss_winner, toss_decision, venue]],
                            columns=['team1','team2','toss_winner','toss_decision','venue'])

    # Encode input
    from sklearn.preprocessing import LabelEncoder
    df_encoded = df_input.copy()
    for col in df_input.columns:
        encoder = joblib.load(f'{col}_encoder.pkl')
        df_encoded[col] = encoder.transform(df_input[col])

    # Predict
    pred = model.predict(df_encoded)[0]
    winner_encoder = joblib.load('winner_encoder.pkl')
    winner = winner_encoder.inverse_transform([pred])[0]

    st.success(f"🏆 Predicted Winner: **{winner}**")