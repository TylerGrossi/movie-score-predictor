"""import streamlit as st
import pandas as pd
import numpy as np

# 🎬 **Title**
st.title("🎬 Movie Score Predictor")

# **Default File Path**
DEFAULT_FILE_PATH = "Movies Ranks.xlsm"

# **Function to Load Movie Data**
def load_movie_data(file_path):
    xls = pd.ExcelFile(file_path)
    return pd.read_excel(xls, sheet_name="Movie Rankings")

# **Load the default file initially**
df_movies = load_movie_data(DEFAULT_FILE_PATH)

# **Optional File Upload**
uploaded_file = st.file_uploader("Upload a new Movies Ranks Excel file (Optional)", type=["xlsm", "xlsx"])

if uploaded_file is not None:
    df_movies = load_movie_data(uploaded_file)
    st.success("✅ New file uploaded successfully!")

# **Compute Average Ratings for Genres and Actors**
genre_columns = ["Genre1", "Genre2", "Genre3"]
actor_columns = ["Actor", "Actor 2", "Actor 3", "Actor 4"]

genre_scores = {}
actor_scores = {}

for _, row in df_movies.iterrows():
    for genre in row[genre_columns]:
        if genre and genre != "None":
            genre_scores.setdefault(genre, []).append(row["Score"])
    for actor in row[actor_columns]:
        if actor and actor != "None":
            actor_scores.setdefault(actor, []).append(row["Score"])

genre_weights = {genre: np.mean(scores) for genre, scores in genre_scores.items()}
actor_avg_rating = {actor: np.mean(scores) for actor, scores in actor_scores.items()}

# **User Inputs**
imdb = st.number_input("Enter IMDb Rating", 0.0, 10.0, 5.0)
rt = st.number_input("Enter Rotten Tomatoes Score", 0, 100, 50)
year = st.number_input("Enter Release Year", 1900, 2025, 2022)

selected_genres = st.multiselect("Select Genres", list(genre_weights.keys()))
selected_actors = st.multiselect("Select Actors", list(actor_avg_rating.keys()))

# **Prediction Function**
if st.button("Predict Score"):
    predicted_score = (imdb + rt / 10 + np.mean([genre_weights.get(g, 5) for g in selected_genres]) + 
                      np.mean([actor_avg_rating.get(a, 5) for a in selected_actors])) / 4
    st.success(f"Predicted Movie Score: {round(predicted_score, 2)} 🎥")
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import csv
from collections import defaultdict, Counter

# Load movie data
file_path = "Movies_Ranks.xlsm"
xls = pd.ExcelFile(file_path)
df_movies = pd.read_excel(xls, sheet_name="Movie Rankings")

# Extract genre and actor columns
genre_columns = ["Genre1", "Genre2", "Genre3"]
actor_columns = ["Actor", "Actor 2", "Actor 3", "Actor 4"]

df_movies[genre_columns] = df_movies[genre_columns].astype(str).replace("nan", None)
df_movies[actor_columns] = df_movies[actor_columns].astype(str).replace("nan", None)
df_movies["Genres"] = df_movies[genre_columns].apply(lambda x: ', '.join(x.dropna()), axis=1)

# Compute average ratings for genres and actors
genre_scores = defaultdict(list)
actor_scores = defaultdict(list)

for _, row in df_movies.iterrows():
    for genre in row[genre_columns]:
        if genre and genre != "None":
            genre_scores[genre].append(row["Score"])
    for actor in row[actor_columns]:
        if actor and actor != "None":
            actor_scores[actor].append(row["Score"])

# Create dynamic weights
genre_weights = {genre: sum(scores) / len(scores) for genre, scores in genre_scores.items()}
actor_avg_rating = {actor: sum(scores) / len(scores) for actor, scores in actor_scores.items()}

# List of unique genres and frequent actors
all_genres = sorted(genre_weights.keys())
actor_counts = Counter(df_movies['Actor'].dropna().tolist() + df_movies['Actor 2'].dropna().tolist())
frequent_actors = sorted([actor for actor, count in actor_counts.items() if count >= 2])

# Boost factors
genre_boost_factor = 1.9
actor_boost_factor = 1.5
year_decay_factor = 0.01

def apply_year_boost(year):
    latest_year = max(df_movies["Year"].dropna())
    return 1 - (latest_year - year) * year_decay_factor

def encode_genres_boosted(selected_genres):
    valid_genres = [genre for genre in selected_genres if genre in genre_weights]
    if not valid_genres:
        return np.mean(list(genre_weights.values()))
    encoded = [genre_weights.get(genre, np.mean(list(genre_weights.values()))) * genre_boost_factor for genre in valid_genres]
    return np.mean(encoded)

def get_actor_rating_boosted(actor):
    if actor not in actor_avg_rating:
        return np.mean(list(actor_avg_rating.values()))
    return actor_avg_rating[actor] * actor_boost_factor

# File to store predictions
prediction_file = "Predicted_Movie_Scores.csv"

def save_prediction(movie_name, predicted_score, actor1, actor2, imdb, rt, year, genres):
    file_exists = os.path.isfile(prediction_file)
    with open(prediction_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Movie Name", "Predicted Score", "Actor 1", "Actor 2", "IMDb Rating", "Rotten Tomatoes Score", "Year", "Genres"])
        writer.writerow([movie_name, predicted_score, actor1, actor2, imdb, rt, year, ", ".join(genres)])

def load_predictions():
    if os.path.isfile(prediction_file):
        return pd.read_csv(prediction_file)
    return pd.DataFrame(columns=["Movie Name", "Predicted Score", "Actor 1", "Actor 2", "IMDb Rating", "Rotten Tomatoes Score", "Year", "Genres"])

st.title("Movie Score Predictor")
tabs = st.tabs(["Predict Score", "Saved Predictions"])

with tabs[0]:
    st.header("Enter Movie Details")
    imdb = st.number_input("IMDb Rating", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    rt = st.number_input("Rotten Tomatoes Score", min_value=0, max_value=100, value=50, step=1)
    year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2024, step=1)
    selected_genres = st.multiselect("Select Up to 3 Genres", all_genres, max_selections=3)
    actor1 = st.selectbox("Lead Actor 1", [None] + frequent_actors)
    actor2 = st.selectbox("Lead Actor 2", [None] + frequent_actors)
    movie_name = st.text_input("Movie Name")

    if st.button("Predict & Save Score"):
        predicted_score = round((imdb + rt / 10 + encode_genres_boosted(selected_genres) + np.mean([get_actor_rating_boosted(actor1), get_actor_rating_boosted(actor2)]) + apply_year_boost(year)) / 5, 2)
        if movie_name:
            save_prediction(movie_name, predicted_score, actor1, actor2, imdb, rt, year, selected_genres)
            st.success(f"Predicted Score: {predicted_score} - Saved Successfully!")
        else:
            st.warning("Please enter a movie name to save the prediction.")

with tabs[1]:
    st.header("Saved Predictions")
    predictions_df = load_predictions()
    st.dataframe(predictions_df)
