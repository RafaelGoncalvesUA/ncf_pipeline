# SERVING THE DL MODEL USING A FLASK API

import torch
from flask import Flask, request, jsonify

import sys
sys.path.append("pipeline")
from pipeline.best_model import NonLinearModel

app = Flask(__name__)

# LOAD MODEL
model = NonLinearModel(162541, 59047, 512, 0.2)
model.load_state_dict(torch.load("models/non_linear.pth"))

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     user_id = data["userId"]
#     item_id = data["itemId"]
#     rating = predict(model, user_id, item_id, "cpu")
#     return jsonify({"rating": rating})

# app.run(port=5000)

"""
COMMAND TO TEST THE API:
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"userId": 1, "itemId": 1}'
"""