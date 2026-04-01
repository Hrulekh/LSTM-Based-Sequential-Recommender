from flask import Flask, request, jsonify
import torch
import random

from models.model import LSTMRecommender
from utils.recommend import (
    get_next_watch,
    get_top_recommendations,
    get_movie_titles,
    movie_map
)

app = Flask(__name__)

# Number of movies (must match training)
n_movies = len(movie_map)

# Load trained model
seq_model = LSTMRecommender(n_movies, emb_dim=64, hidden_dim=128)
seq_model.load_state_dict(torch.load("notebooks/lstm_model.pth"))
seq_model.eval()


# Home route (so browser works)
@app.route("/")
def home():
    return "Movie Recommendation API is running 🚀"


# Main recommendation route
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    
    # If opened in browser (GET request)
    if request.method == "GET":
        return "Use POST request with JSON: {'history': [10,25,40]}"

    # POST request
    data = request.get_json()
    user_history = data.get("history", [])

    # Cold start (new user)
    if len(user_history) < 3:
        random_movies = random.sample(range(n_movies), 10)
        return jsonify({
            "next_watch": get_movie_titles(random_movies),
            "top_recommendations": get_movie_titles(random_movies)
        })

    # Normal recommendation
    next_watch = get_next_watch(seq_model, user_history)
    top_recs = get_top_recommendations(seq_model, user_history)

    return jsonify({
        "next_watch": get_movie_titles(next_watch),
        "top_recommendations": get_movie_titles(top_recs)
    })


if __name__ == "__main__":
    app.run(debug=True)