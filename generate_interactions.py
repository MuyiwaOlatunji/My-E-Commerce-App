# generate_interactions.py (same as before)
import sqlite3
import random
from datetime import datetime, timedelta

def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db_connection()
c = conn.cursor()
c.execute('SELECT id FROM users')
users = [row['id'] for row in c.fetchall()]
c.execute('SELECT id FROM products')
products = [row['id'] for row in c.fetchall()]
c.execute('DELETE FROM interactions')
actions = ['view', 'buy']
interactions = []

for user_id in users:
    num_interactions = random.randint(1, 5)
    interacted_products = random.sample(products, num_interactions)
    for product_id in interacted_products:
        action = random.choice(actions)
        days_ago = random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        interactions.append((user_id, product_id, action, timestamp))

c.executemany('INSERT INTO interactions (user_id, product_id, action, timestamp) VALUES (?, ?, ?, ?)', interactions)
conn.commit()
c.execute('SELECT * FROM interactions LIMIT 5')
print("Sample interactions:", [dict(row) for row in c.fetchall()])
conn.close()
print(f"Generated {len(interactions)} interactions.")
