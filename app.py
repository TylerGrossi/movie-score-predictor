import streamlit as st
import pandas as pd
import numpy as np

# 🎬 **Title**
st.title("🎬 Movie Score Predictor")

# **Default File Path**
DEFAULT_FILE_PATH = "/mnt/data/Movies Ranks.xlsm"

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
