# generate_interactions.py
import sqlite3
import random
import os
from datetime import datetime, timedelta

def get_db_connection():
    try:
        app_data_dir = os.path.join(os.getenv('APPDATA'), 'EcommerceApp')
        os.makedirs(app_data_dir, exist_ok=True)
        db_path = os.path.join(app_data_dir, 'ecommerce.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

conn = get_db_connection()
if conn is None:
    raise Exception("Failed to connect to the database")
c = conn.cursor()

# Fetch users and products
c.execute('SELECT id FROM users')
users = [row['id'] for row in c.fetchall()]
c.execute('SELECT id FROM products')
products = [row['id'] for row in c.fetchall()]

# Clear existing interactions
c.execute('DELETE FROM interactions')

# Generate new interactions
actions = ['view', 'buy']
interactions = []

if not users or not products:
    print("No users or products found in the database. Please ensure the database is populated.")
else:
    for user_id in users:
     num_interactions = random.randint(5, 10)  # Increase from 1-5 to 5-10
     interacted_products = random.sample(products, min(num_interactions, len(products)))
     for product_id in interacted_products:
        action = random.choice(actions)
        days_ago = random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        interactions.append((user_id, product_id, action, timestamp))

    c.executemany('INSERT INTO interactions (user_id, product_id, action, timestamp) VALUES (?, ?, ?, ?)', interactions)
    conn.commit()

    # Print sample interactions
    c.execute('SELECT * FROM interactions LIMIT 5')
    print("Sample interactions:", [dict(row) for row in c.fetchall()])
    print(f"Generated {len(interactions)} interactions.")

conn.close()