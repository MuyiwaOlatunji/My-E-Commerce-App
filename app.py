# app.py
import os
import psycopg2
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import sqlite3
from datetime import datetime
import random
from collections import defaultdict

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '5507e4842f2c53c15f4a3bbd1e004e6ef59eb7007920c29d1c2b1bc133d90336') # Replace with a random secret key for production

#DBMS Initialization
def get_db_connection():
    conn = psycopg2.connect(os.environ.get('DATABSE_URL')) # Replace with your database URL
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, 
                  preferences TEXT)''')
    
    # Products table
    c.execute('''CREATE TABLE IF NOT EXISTS products 
                 (id SERIAL PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL, category TEXT, 
                  stock INTEGER NOT NULL)''')
    
    # Orders table
    c.execute('''CREATE TABLE IF NOT EXISTS orders 
                 (id SERIAL PRIMARY KEY, user_id INTEGER, product_id INTEGER, 
                  quantity INTEGER, timestamp TEXT, status TEXT, FOREIGN KEY (user_id) REFERENCES users(id),
                  FOREIGN KEY (product_id) REFERENCES products(id))''')
    
    # New Cart table
    c.execute('''CREATE TABLE IF NOT EXISTS cart 
                 (id SERIAL PRIMARY KEY, user_id INTEGER, product_id INTEGER, 
                  quantity INTEGER, FOREIGN KEY (user_id) REFERENCES users(id),
                  FOREIGN KEY (product_id) REFERENCES products(id))''')
    
    # Sample products (Distribution)
    sample_products = [
        (1, 'Wireless Mouse', 29.99, 'Electronics', 100),
        (2, 'Graphic T-shirt', 15.99, 'Clothing', 200),
        (3, 'Bluetooth Headphones', 49.99, 'Electronics', 150),
        (4, 'Coffee Maker', 89.99, 'Home Appliances', 50),
        (5, 'Smartphone', 699.99, 'Electronics', 30),
        (6, 'Running Shoes', 59.99, 'Footwear', 80),
        (7, 'Backpack', 39.99, 'Accessories', 120),
        (8, 'Smartwatch', 199.99, 'Electronics', 70),
        (9, 'Yoga Mat', 25.99, 'Fitness', 90),
        (10, 'Wireless Charger', 19.99, 'Electronics', 110),
        (11, 'Wireless Keyboard', 29.99, 'Electronics', 100),
        (12, 'Rockstar T-shirt', 15.99, 'Clothing', 200),
        (13, 'Python Book', 24.99, 'Books', 50),
        (14, 'Earpods', 59.99, 'Electronics', 75),
        (15, 'Jeans', 39.99, 'Clothing', 150)
        ]
    c.executemany('INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)', sample_products)

    conn.commit()
    conn.close()
    print("Database initialized successfully")

# Cart Functionality
def get_cart_count():
    if 'user_id' not in session:
        return 0
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    count = c.execute('SELECT SUM(quantity) FROM cart WHERE user_id=?',
                     (session['user_id'],)).fetchone()[0]
    conn.close()
    return count or 0

# System Authentication
@app.route('/register', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        preferences = request.form.get('preferences', '') #Personal Characteristics

        conn = sqlite3.connect('ecommerce.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password, preferences) VALUES (?, ?, ?)', (username, password, preferences))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return 'Username already exists'
        finally:
            conn.close()
    return render_template('register.html', cart_count = 0)
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('ecommerce.db')
        c = conn.cursor()
        user = c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password)).fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('home'))
        return "Invalid Credentials"
    return render_template('login.html', cart_count = 0)

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    
    # Get cart items with product details
    cart_items = c.execute('''
        SELECT p.id, p.name, p.price, p.category, c.quantity, p.stock
        FROM cart c
        JOIN products p ON c.product_id = p.id
        WHERE c.user_id=?
    ''', (session['user_id'],)).fetchall()
    
    conn.close()
    
    cart_count = get_cart_count()
    return render_template('cart.html', cart_items=cart_items, cart_count=cart_count)

# Cart Tracker
@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    
    # Get cart items
    cart_items = c.execute('''
        SELECT c.product_id, c.quantity, p.stock, p.name
        FROM cart c
        JOIN products p ON c.product_id = p.id
        WHERE c.user_id=?
    ''', (session['user_id'],)).fetchall()
    
    
    if not cart_items:
        conn.close()
        return jsonify({'success': False, 'message': 'Your cart is empty'}), 400
    
    # Check stock availability
    for item in cart_items:
        product_id, quantity, stock, name = item
        if quantity > stock:
            conn.close()
            return jsonify({'success': False, 'message': f'Not enough stock for {name}'}), 400
    
    # Create orders and update stock
    try:
        timestamp = datetime.now().isoformat()
        for item in cart_items:
            product_id, quantity, _, _ = item
            # Insert into orders
            c.execute('INSERT INTO orders (user_id, product_id, quantity, timestamp, status) VALUES (?,?,?,?,?)',
                     (session['user_id'], product_id, quantity, timestamp, 'Pending'))
            # Update stock
            c.execute('UPDATE products SET stock = stock - ? WHERE id=?', (quantity, product_id))
        
        # Clear cart
        c.execute('DELETE FROM cart WHERE user_id=?', (session['user_id'],))
        
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Checkout successful! Your order has been placed.'})
    
    except sqlite3.Error as e:
        conn.rollback()
        conn.close()
        return jsonify({'success': False, 'message': 'An error occurred during checkout'}), 500
    

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
    
# E-commerce functionality
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    # Verify user exists
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    user = c.execute('SELECT id FROM users WHERE id=?', (session['user_id'],)).fetchone()
    
    if user is None:
        session.pop('user_id', None)
        conn.close()
        return redirect(url_for('login'))
    
    products = c.execute('SELECT * FROM products WHERE stock > 0').fetchall()
    recommendations = get_personalized_recommendations(session['user_id'])
    
    conn.close()
    
    cart_count = get_cart_count()
    return render_template('home.html', products=products, recommendations=recommendations,
    cart_count=cart_count)


@app.route('/buy/<int:product_id>', methods=['POST'])
def buy(product_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    quantity = int(request.form.get('quantity', 1))
    
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    
    # Check stock availability
    product = c.execute('SELECT stock, name FROM products WHERE id=?', (product_id,)).fetchone()
    if not product:
        conn.close()
        return jsonify({'success': False, 'message': 'Product not found'}), 404
    
    if product[0] < quantity:
        conn.close()
        return jsonify({'success': False, 'message': 'Not enough stock'}), 400
    
    # Check if item is already in cart
    cart_item = c.execute('SELECT quantity FROM cart WHERE user_id=? AND product_id=?',
                         (session['user_id'], product_id)).fetchone()
    
    if cart_item:
        # Update quantity
        new_quantity = cart_item[0] + quantity
        c.execute('UPDATE cart SET quantity=? WHERE user_id=? AND product_id=?',
                 (new_quantity, session['user_id'], product_id))
    else:
        # Add new cart item
        c.execute('INSERT INTO cart (user_id, product_id, quantity) VALUES (?,?,?)',
                 (session['user_id'], product_id, quantity))
    
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'message': f'{product[1]} added to cart!',
        'product_id': product_id,
        'quantity': quantity
    })

# Personalized Recommendation System
def get_personalized_recommendations(user_id):
    conn = sqlite3.connect('ecommerce.db')
    c = conn.cursor()
    
    # Get user preferences and purchase history
    user_data = c.execute('SELECT preferences FROM users WHERE id=?', (user_id,)).fetchone()
    
    # Handle case where user is not found
    if user_data is None:
        preferences = []
    else:
        preferences = user_data[0].split(',') if user_data[0] else []
    
    orders = c.execute('SELECT product_id FROM orders WHERE user_id=?', (user_id,)).fetchall()
    
    purchased_products = set(order[0] for order in orders)
    
    # Simple recommendation logic
    rec_products = []
    
    if preferences:
        placeholders = ','.join('?' for _ in preferences)
        rec_products = c.execute(
            f'SELECT * FROM products WHERE category IN ({placeholders}) AND stock > 0 AND id NOT IN ({",".join("?" for _ in purchased_products)})',
            preferences + list(purchased_products)
        ).fetchall()
    
    # If less than 2 recommendations, add popular items
    if len(rec_products) < 2:
        popular = c.execute('''
            SELECT p.*, COUNT(o.product_id) as order_count 
            FROM products p 
            LEFT JOIN orders o ON p.id = o.product_id 
            WHERE p.stock > 0 AND p.id NOT IN ({})
            GROUP BY p.id 
            ORDER BY order_count DESC 
            LIMIT 2'''.format(','.join('?' for _ in purchased_products)),
            list(purchased_products)
        ).fetchall()
        rec_products.extend(popular)
    
    conn.close()
    return rec_products[:2]

 # Return top 2 recommendations

# Run App
if __name__ == '__main__':
    init_db()
    app.run(debug = True, ssl_context = 'adhoc')