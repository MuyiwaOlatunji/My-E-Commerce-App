import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2  # Concatenated user and item embeddings
        for layer_size in layers:
            self.fc_layers.append(nn.Linear(input_size, layer_size))
            input_size = layer_size
        self.output_layer = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        x = torch.cat([user_embeds, item_embeds], dim=-1)
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return self.sigmoid(x)