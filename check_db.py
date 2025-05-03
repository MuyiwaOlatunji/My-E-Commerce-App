# check_db.py
import sqlite3
import os

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

# Check users
c.execute('SELECT * FROM users')
users = c.fetchall()
print("Users in database:", [dict(row) for row in users])

# Check products
c.execute('SELECT * FROM products LIMIT 5')
products = c.fetchall()
print("Sample products in database:", [dict(row) for row in products])

# Check total counts
c.execute('SELECT COUNT(*) FROM users')
user_count = c.fetchone()[0]
c.execute('SELECT COUNT(*) FROM products')
product_count = c.fetchone()[0]
print(f"Total users: {user_count}, Total products: {product_count}")

conn.close()