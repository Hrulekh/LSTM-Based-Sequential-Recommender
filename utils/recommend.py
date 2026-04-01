import torch
import pandas as pd
import pickle

# Load movie_map
with open("notebooks/movie_map.pkl", "rb") as f:
    movie_map = pickle.load(f)

idx_to_movie = {v: k for k, v in movie_map.items()}

# Load movies dataset
movies = pd.read_csv(
    "data/ml-1m/movies.dat",
    sep="::",
    engine="python",
    names=["movie","title","genre"],
    encoding="latin-1"
)

# Convert indices → titles
def get_movie_titles(indices):
    movie_ids = [idx_to_movie[i] for i in indices]
    id_to_title = dict(zip(movies["movie"], movies["title"]))
    return [id_to_title[mid] for mid in movie_ids]


# Next watch (direct prediction)
def get_next_watch(model, sequence, k=10):
    seq_tensor = torch.tensor(sequence).unsqueeze(0)
    
    with torch.no_grad():
        scores = model(seq_tensor).squeeze()
    
    return torch.topk(scores, k).indices.tolist()


# Top recommendations (diversified)
def get_top_recommendations(model, user_history, k=10):
    sequence = user_history[-3:]
    
    seq_tensor = torch.tensor(sequence).unsqueeze(0)
    
    with torch.no_grad():
        scores = model(seq_tensor).squeeze()
    
    watched = set(user_history)
    
    candidates = []
    for i, s in enumerate(scores):
        if i not in watched:
            candidates.append((i, s.item()))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # skip top 5 for diversity
    filtered = candidates[5:5+k]
    
    return [x[0] for x in filtered]