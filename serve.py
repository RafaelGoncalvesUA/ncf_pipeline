# serve a model
import torch
from flask import Flask, request, jsonify

# select model with imports

class CollaborativeFilteringModel(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        self.fc1 = torch.nn.Linear(embedding_dim * 2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, user_ids, item_ids):
        if torch.any(user_ids >= self.user_embedding.num_embeddings):
            raise ValueError(f"user_ids contain indices outside the range: {user_ids} | {self.user_embedding.num_embeddings}")
        if torch.any(item_ids >= self.item_embedding.num_embeddings):
            raise ValueError(f"item_ids contain indices outside the range: {item_ids} | {self.item_embedding.num_embeddings}")

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, user_id, item_id, device):
        self.eval()
        with torch.no_grad():
            user_tensor = torch.tensor([user_id]).to(device)
            item_tensor = torch.tensor([item_id]).to(device)
            rating = self(user_tensor, item_tensor)
            return rating.item()


app = Flask(__name__)

NUM_USERS = 162541
NUM_MOVIES = 59047
EMBEDDING_DIM = 10
model = CollaborativeFilteringModel(NUM_USERS, NUM_MOVIES, EMBEDDING_DIM)
# LOAD MODEL
model.load_state_dict(torch.load("models/non_linear.pth"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_id = data["userId"]
    item_id = data["itemId"]
    rating = model.predict(user_id, item_id, "cpu")
    return jsonify({"rating": rating})

app.run(port=5000)

# CURL COMMAND TO TEST
"""
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"userId": 1, "itemId": 1}'
"""