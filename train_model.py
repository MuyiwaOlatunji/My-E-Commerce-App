import torch
import torch.nn as nn
import sqlite3
import numpy as np
from model import NCF

def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn

# Load interaction data
conn = get_db_connection()
c = conn.cursor()
c.execute('SELECT COUNT(DISTINCT user_id) FROM interactions')
num_users = c.fetchone()[0] or 1
c.execute('SELECT COUNT(DISTINCT product_id) FROM interactions')
num_items = c.fetchone()[0] or 1
c.execute('SELECT user_id, product_id, action FROM interactions')
interactions = c.fetchall()
conn.close()

# Prepare training data
user_ids = []
item_ids = []
ratings = []
for interaction in interactions:
    user_id = interaction['user_id']
    product_id = interaction['product_id']
    action = interaction['action']
    rating = 1 if action in ['view', 'buy'] else 0
    user_ids.append(user_id - 1)  # Adjust user_id to 0-indexed
    item_ids.append(product_id - 1)  # Adjust product_id to 0-indexed
    ratings.append(rating)

user_ids = torch.tensor(user_ids, dtype=torch.long)
item_ids = torch.tensor(item_ids, dtype=torch.long)
ratings = torch.tensor(ratings, dtype=torch.float)

# Initialize model
model = NCF(num_users, num_items, embedding_dim=64, layers=[128, 64, 32])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(user_ids, item_ids).squeeze()
    loss = criterion(outputs, ratings)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model trained and saved as model.pth")

