import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import traceback
import sqlite3
from io import StringIO

from main import interactions

# Initialize SQLite database
conn = sqlite3.connect("scores.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS scores (
    team_name TEXT,
    team_lead_email TEXT,
    score REAL,
    submission_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Precompute data using Streamlit cache

# Custom exception class
class ParticipantVisibleError(Exception):
    pass

# Precompute state distribution
def calculate_state_distribution(interactions: pd.DataFrame, users: pd.DataFrame) -> dict:
    merged = pd.merge(interactions, users, left_on='Member_ID', right_on='Member_ID', how='inner')
    merged = pd.merge(merged, users, left_on='Target_ID', right_on='Member_ID', suffixes=('_sender', '_receiver'), how='inner')
    state_dist = merged.groupby('State_sender')['State_receiver'].value_counts(normalize=True).unstack(fill_value=0)
    return state_dist.to_dict(orient='index')

# Precompute caste distribution
def calculate_caste_distribution(interactions: pd.DataFrame, users: pd.DataFrame) -> dict:
    merged = pd.merge(interactions, users, left_on='Member_ID', right_on='Member_ID', how='inner')
    merged = pd.merge(merged, users, left_on='Target_ID', right_on='Member_ID', suffixes=('_sender', '_receiver'), how='inner')
    caste_dist = merged.groupby('Caste_sender')['Caste_receiver'].value_counts(normalize=True).unstack(fill_value=0)
    return caste_dist.to_dict(orient='index')

# Precompute interacted and below-average interacted users
def precompute_diversity_data(users: pd.DataFrame) -> tuple:
    interacted_users = set(users[users['Interaction_Count'] > 0]['Member_ID'])
    avg_interactions = users['Interaction_Count'].mean()
    below_avg_interacted_users = set(users[users['Interaction_Count'] < avg_interactions]['Member_ID'])
    return interacted_users, below_avg_interacted_users

# Check cross-gender recommendations
def check_cross_gender_recommendations(member_id: int, top_100_profiles: list, users: pd.DataFrame) -> bool:
    member_gender = users.loc[users['Member_ID'] == member_id, 'Gender'].iloc[0]
    recommended_genders = users.loc[users['Member_ID'].isin(top_100_profiles), 'Gender']
    if member_gender == 'Male':
        return all(gender == 'Female' for gender in recommended_genders)
    elif member_gender == 'Female':
        return all(gender == 'Male' for gender in recommended_genders)
    return False

# Evaluate diversity using precomputed data
def evaluate_diversity(top_100_profiles: list) -> float:
    diverse_count = sum(1 for profile in top_100_profiles if profile not in interacted_users)
    below_avg_count = sum(1 for profile in top_100_profiles if profile in below_avg_interacted_users)
    diversity_score = (diverse_count / len(top_100_profiles)) * 0.15 + (below_avg_count / len(top_100_profiles)) * 0.05
    return diversity_score

# Match distributions using Euclidean distance
def match_distribution(recommended_distribution: dict, target_distribution: dict) -> float:
    keys = set(recommended_distribution.keys()).union(target_distribution.keys())
    rec_vec = [recommended_distribution.get(key, 0) for key in keys]
    tar_vec = [target_distribution.get(key, 0) for key in keys]
    return 1 - euclidean(rec_vec, tar_vec) / len(keys)

@st.cache_data
def load_and_preprocess_data():
    users = pd.read_csv("users.csv")
    interactions = pd.read_csv("interactions.csv")

    # Precompute interaction counts
    interaction_counts = interactions.groupby('Member_ID').size().reset_index(name='Interaction_Count')
    users = pd.merge(users, interaction_counts, on='Member_ID', how='left').fillna(0)
    users['Interaction_Count'] = users['Interaction_Count'].astype(int)

    # Precompute state and caste distributions
    state_distribution = calculate_state_distribution(interactions, users)
    caste_distribution = calculate_caste_distribution(interactions, users)

    # Precompute diversity data
    interacted_users, below_avg_interacted_users = precompute_diversity_data(users)

    return users, interactions, state_distribution, caste_distribution, interacted_users, below_avg_interacted_users

# Load and preprocess data once
users, interactions, state_distribution, caste_distribution, interacted_users, below_avg_interacted_users = load_and_preprocess_data()

# Scoring function
def score(submission: pd.DataFrame, row_id_column_name: str, users: pd.DataFrame) -> float:
    total_score = 0.0
    num_rows = len(submission)

    for _, row in submission.iterrows():
        member_id = row['Member_ID']
        top_100_profiles = [int(x.strip()) for x in row['Top_100_Profiles'].split(',')]

        # Cross-gender check
        if not check_cross_gender_recommendations(member_id, top_100_profiles, users):
            raise ParticipantVisibleError(f"Recommendations for Member_ID {member_id} are not cross-gender.")

        # Interaction check
        sender_interactions = interactions[interactions['Member_ID'] == member_id]
        if not sender_interactions.empty:
            if any(profile in sender_interactions['Target_ID'].values for profile in top_100_profiles):
                raise ParticipantVisibleError(f"Duplicate interaction found for Member_ID {member_id}.")

        # Sect match
        sect_match_score = sum(1 for profile in top_100_profiles if users.loc[users['Member_ID'] == profile, 'Sect'].iloc[0] == users.loc[users['Member_ID'] == member_id, 'Sect'].iloc[0]) / len(top_100_profiles) * 0.2

        # Marital status match
        marital_status_match_score = sum(1 for profile in top_100_profiles if users.loc[users['Member_ID'] == profile, 'Marital_Status'].iloc[0] == users.loc[users['Member_ID'] == member_id, 'Marital_Status'].iloc[0]) / len(top_100_profiles) * 0.2

        # Age condition
        avg_age = users.loc[users['Member_ID'].isin(top_100_profiles), 'Age'].mean()
        member_age = users.loc[users['Member_ID'] == member_id, 'Age'].iloc[0]
        age_condition_score = 0.0
        if users.loc[users['Member_ID'] == member_id, 'Gender'].iloc[0] == 'Male':
            if avg_age < member_age:
                age_condition_score += 0.1
            top_5_ages = users.loc[users['Member_ID'].isin(top_100_profiles[:5]), 'Age']
            if all(age < member_age for age in top_5_ages):
                age_condition_score += 0.1
        else:
            if avg_age > member_age:
                age_condition_score += 0.1
            top_5_ages = users.loc[users['Member_ID'].isin(top_100_profiles[:5]), 'Age']
            if all(age > member_age for age in top_5_ages):
                age_condition_score += 0.1

        # State distribution match
        sender_state = users.loc[users['Member_ID'] == member_id, 'State'].iloc[0]
        recommended_states = users.loc[users['Member_ID'].isin(top_100_profiles), 'State'].value_counts(normalize=True).to_dict()
        state_distribution_score = match_distribution(recommended_states, state_distribution.get(sender_state, {})) * 0.2

        # Caste distribution match
        sender_caste = users.loc[users['Member_ID'] == member_id, 'Caste'].iloc[0]
        recommended_castes = users.loc[users['Member_ID'].isin(top_100_profiles), 'Caste'].value_counts(normalize=True).to_dict()
        caste_distribution_score = match_distribution(recommended_castes, caste_distribution.get(sender_caste, {})) * 0.2

        # Diversity
        diversity_score = evaluate_diversity(top_100_profiles)

        # Total score for this row
        row_score = sect_match_score + marital_status_match_score + age_condition_score + state_distribution_score + caste_distribution_score + diversity_score
        total_score += row_score

    return total_score / num_rows

# Streamlit App
st.title("Matrimonial Recommendation Scoring System")

# Input fields for team details
team_name = st.text_input("Enter Team Name:")
team_lead_email = st.text_input("Enter Team Lead Email:")

# File uploader for submission CSV
submission_file = st.file_uploader("Upload Submission CSV", type=["csv"])

# Button to calculate score
if st.button("Calculate Score"):
    if not team_name or not team_lead_email:
        st.error("Please provide both Team Name and Team Lead Email.")
    elif not submission_file:
        st.error("Please upload Submission CSV files.")
    else:
        try:
            # Display processing message
            with st.spinner("Processing your submission... This may take up to 10 minutes."):
                # Read uploaded CSV file
                submission = pd.read_csv(StringIO(submission_file.getvalue().decode("utf-8")))

                # Validate required columns
                required_columns = {"Member_ID", "Top_100_Profiles"}
                if not required_columns.issubset(submission.columns):
                    raise ParticipantVisibleError("Submission CSV must contain 'Member_ID' and 'Top_100_Profiles' columns.")

                # Calculate score
                accuracy = score(submission, "Member_ID", users)

                # Save to database
                cursor.execute("INSERT INTO scores (team_name, team_lead_email, score) VALUES (?, ?, ?)", (team_name, team_lead_email, accuracy))
                conn.commit()

                # Display results
                st.success(f"Team Name: {team_name}")
                st.success(f"Team Lead Email: {team_lead_email}")
                st.success(f"Accuracy: {accuracy:.4f}")

        except ParticipantVisibleError as e:
            st.error(f"Error: {str(e)}")
            st.error("Score: 0.0")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")
            st.error("Score: 0.0")
            st.text("Traceback:")
            st.text(traceback.format_exc())