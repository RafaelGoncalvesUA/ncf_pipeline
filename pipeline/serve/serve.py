# SERVING THE DL MODEL USING A FLASK API

import torch
from flask import Flask, request, jsonify
import pickle

import sys
sys.path.append("pipeline")

from best_model import NonLinearModel

NUM_USERS = 162541
NUM_MOVIES = 59047
EMBEDDING_DIM = 512
DROPOUT = 0.2

app = Flask(__name__)

# LOAD MODEL
model = NonLinearModel(NUM_USERS, NUM_MOVIES, 512, 0.2)
model.load_state_dict(torch.load("pipeline/model.pth")) # run in root directory

# load user2idx / item2idx mappings and scaler
user2idx = pickle.load(open("pipeline/user2idx.pkl", "rb"))
item2idx = pickle.load(open("pipeline/item2idx.pkl", "rb"))
scaler = pickle.load(open("pipeline/scaler.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_id = user2idx[data["userId"]]
    item_id = item2idx[data["itemId"]]
    scaled_rating = model.predict(user_id, item_id, device)
    rating = scaler.inverse_transform([[scaled_rating]])[0][0]
    return jsonify({"rating": rating})

app.run(port=5000)

"""
COMMAND TO TEST THE API:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"userId": 1, "itemId": 1}'
"""