# train_ncf_model.py
import os
import sqlite3
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from datetime import datetime

# Define the NCFModel class (same as in app.py)
class NCFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        x = self.fc_layers(concat)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Database connection and initialization
def get_db_connection():
    try:
        app_data_dir = os.path.join(os.getenv('APPDATA'), 'EcommerceApp')
        os.makedirs(app_data_dir, exist_ok=True)  # Create the directory if it doesn't exist
        db_path = os.path.join(app_data_dir, 'ecommerce.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        raise Exception("Failed to connect to the database")
    try:
        c = conn.cursor()
        
        # Create tables (same as in app.py)
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                preferences TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT,
                stock INTEGER NOT NULL,
                image TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                total_price REAL,
                timestamp TEXT,
                status TEXT,
                payment_id TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS order_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                price REAL,
                FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS cart (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                user_id INTEGER,
                product_id INTEGER,
                action TEXT,
                timestamp DATETIME,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        # Insert sample products if none exist
        c.execute('SELECT COUNT(*) FROM products')
        if c.fetchone()[0] == 0:
            sample_products = [
                ('Wireless Mouse', 29.99, 'Electronics', 100, '/static/images/wireless_mouse.jpg'),
                ('Graphic T-shirt', 15.99, 'Clothing', 200, '/static/images/graphic_t-shirt.jpg'),
                ('Bluetooth Headphones', 49.99, 'Electronics', 150, '/static/images/bluetooth_headphones.jpg'),
                ('Coffee Maker', 89.99, 'Home Appliances', 50, '/static/images/coffee_maker.jpg'),
                ('Smartphone', 699.99, 'Electronics', 30, '/static/images/smartphone.jpg'),
                ('Laptop Stand', 39.99, 'Electronics', 75, '/static/images/laptop_stand.jpg'),
                ('Running Shoes', 59.99, 'Clothing', 120, '/static/images/running_shoes.jpg'),
                ('Desk Lamp', 24.99, 'Home Appliances', 90, '/static/images/desk_lamp.jpg'),
                ('USB-C Cable', 9.99, 'Electronics', 200, '/static/images/usb-c_cable.jpg'),
                ('Winter Jacket', 89.99, 'Clothing', 60, '/static/images/winter_jacket.jpg'),
                ('Electric Kettle', 34.99, 'Home Appliances', 70, '/static/images/electric_kettle.jpg'),
                ('Gaming Keyboard', 69.99, 'Electronics', 50, '/static/images/gaming_keyboard.jpg'),
                ('Sweatshirt', 29.99, 'Clothing', 150, '/static/images/sweatshirt.jpg'),
                ('Blender', 49.99, 'Home Appliances', 40, '/static/images/blender.jpg'),
                ('Portable Charger', 19.99, 'Electronics', 100, '/static/images/portable_charger.jpg'),
                ('Jeans', 39.99, 'Clothing', 110, '/static/images/jeans.jpg'),
                ('Air Fryer', 79.99, 'Home Appliances', 30, '/static/images/air_fryer.jpg'),
                ('Smartwatch', 129.99, 'Electronics', 25, '/static/images/smartwatch.jpg'),
                ('Baseball Cap', 14.99, 'Clothing', 180, '/static/images/baseball_cap.jpg'),
                ('Toaster', 29.99, 'Home Appliances', 60, '/static/images/toaster.jpg'),
                ('Wireless Earbuds', 39.99, 'Electronics', 90, '/static/images/wireless_earbuds.jpg'),
                ('Hoodie', 34.99, 'Clothing', 130, '/static/images/hoodie.jpg'),
                ('Microwave Oven', 99.99, 'Home Appliances', 20, '/static/images/microwave_oven.jpg'),
                ('Tablet', 299.99, 'Electronics', 15, '/static/images/tablet.jpg'),
                ('Sneakers', 49.99, 'Clothing', 100, '/static/images/sneakers.jpg'),
                ('Vacuum Cleaner', 149.99, 'Home Appliances', 10, '/static/images/vacuum_cleaner.jpg'),
                ('External Hard Drive', 79.99, 'Electronics', 40, '/static/images/external_hard_drive.jpg'),
                ('Scarf', 12.99, 'Clothing', 200, '/static/images/scarf.jpg'),
                ('Electric Fan', 44.99, 'Home Appliances', 50, '/static/images/electric_fan.jpg'),
                ('Webcam', 59.99, 'Electronics', 30, '/static/images/webcam.jpg'),
                ('Socks (Pack of 5)', 9.99, 'Clothing', 300, '/static/images/socks.jpg'),
                ('Food Processor', 69.99, 'Home Appliances', 25, '/static/images/food_processor.jpg'),
                ('Fitness Tracker', 49.99, 'Electronics', 35, '/static/images/fitness_tracker.jpg'),
                ('Trench Coat', 99.99, 'Clothing', 40, '/static/images/trench_coat.jpg'),
                ('Humidifier', 39.99, 'Home Appliances', 45, '/static/images/humidifier.jpg'),
            ]
            c.executemany('INSERT INTO products (name, price, category, stock, image) VALUES (?, ?, ?, ?, ?)', sample_products)
        
        # Insert sample interactions if none exist
        c.execute('SELECT COUNT(*) FROM interactions')
        if c.fetchone()[0] == 0:
            sample_interactions = [
                (1, 1, 'view', datetime.now().isoformat()),
                (1, 2, 'view', datetime.now().isoformat()),
                (2, 3, 'view', datetime.now().isoformat()),
                (2, 1, 'view', datetime.now().isoformat()),
            ]
            c.executemany('INSERT INTO interactions (user_id, product_id, action, timestamp) VALUES (?, ?, ?, ?)', sample_interactions)
        
        conn.commit()
        print("Database initialized successfully")
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        raise e
    finally:
        conn.close()

# Connect to the database and initialize if needed
init_db()
conn = get_db_connection()
if conn is None:
    raise Exception("Failed to connect to the database after initialization")
conn.row_factory = sqlite3.Row
c = conn.cursor()

# Get the number of users and items
c.execute('SELECT COUNT(DISTINCT user_id) FROM interactions')
num_users = c.fetchone()[0] or 1
c.execute('SELECT COUNT(DISTINCT product_id) FROM interactions')
num_items = c.fetchone()[0] or 1
print(f"Number of users: {num_users}, Number of items: {num_items}")

# Load interaction data
c.execute('SELECT user_id, product_id, action FROM interactions')
interactions = c.fetchall()
conn.close()

# Prepare training data
if not interactions:
    print("No interaction data found. Generating synthetic data...")
    num_samples = 1000
    user_ids = torch.randint(0, max(num_users, 1), (num_samples,))
    item_ids = torch.randint(0, max(num_items, 1), (num_samples,))
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.float)
else:
    print(f"Found {len(interactions)} interactions.")
    user_ids_list = []
    item_ids_list = []
    labels_list = []
    
    for interaction in interactions:
        user_id = interaction['user_id'] - 1  # Adjust for 0-based indexing
        item_id = interaction['product_id'] - 1  # Adjust for 0-based indexing
        label = 1.0 if interaction['action'] == 'view' else 0.0
        
        if user_id < num_users and item_id < num_items:
            user_ids_list.append(user_id)
            item_ids_list.append(item_id)
            labels_list.append(label)
    
    if not user_ids_list:
        print("No valid interactions after filtering. Generating synthetic data...")
        num_samples = 1000
        user_ids = torch.randint(0, max(num_users, 1), (num_samples,))
        item_ids = torch.randint(0, max(num_items, 1), (num_samples,))
        labels = torch.randint(0, 2, (num_samples,), dtype=torch.float)
    else:
        user_ids = torch.tensor(user_ids_list, dtype=torch.long)
        item_ids = torch.tensor(item_ids_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.float)

# Initialize the model
model = NCFModel(num_users, num_items, embedding_dim=64)
optimizer = Adam(model.parameters(), lr=0.0005)
criterion = nn.BCELoss()

# Training loop
model.train()
num_epochs = 50
batch_size = 32
num_samples = len(user_ids)

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, num_samples, batch_size):
        batch_user_ids = user_ids[i:i + batch_size]
        batch_item_ids = item_ids[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_user_ids, batch_item_ids).squeeze()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / (num_samples // batch_size + 1)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")