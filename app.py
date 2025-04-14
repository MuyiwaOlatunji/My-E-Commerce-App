import os
import psycopg2
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from datetime import datetime
import random
from collections import defaultdict

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set. Please set it in environment variables.")

# PostgreSQL Database Connection
def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            preferences TEXT
        )
    ''')
    
    # Products table
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT,
            stock INTEGER NOT NULL
        )
    ''')
    
    # Orders table
    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
            quantity INTEGER,
            timestamp TEXT,
            status TEXT
        )
    ''')
    
    # Cart table
    c.execute('''
        CREATE TABLE IF NOT EXISTS cart (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
            quantity INTEGER
        )
    ''')
    
    # Sample products
    sample_products = [
        ('Wireless Mouse', 29.99, 'Electronics', 100),
        ('Graphic T-shirt', 15.99, 'Clothing', 200),
        ('Bluetooth Headphones', 49.99, 'Electronics', 150),
        ('Coffee Maker', 89.99, 'Home Appliances', 50),
        ('Smartphone', 699.99, 'Electronics', 30),
        ('Running Shoes', 59.99, 'Footwear', 80),
        ('Backpack', 39.99, 'Accessories', 120),
        ('Smartwatch', 199.99, 'Electronics', 70),
        ('Yoga Mat', 25.99, 'Fitness', 90),
        ('Wireless Charger', 19.99, 'Electronics', 110),
        ('Wireless Keyboard', 29.99, 'Electronics', 100),
        ('Rockstar T-shirt', 15.99, 'Clothing', 200),
        ('Python Book', 24.99, 'Books', 50),
        ('Earpods', 59.99, 'Electronics', 75),
        ('Jeans', 39.99, 'Clothing', 150)
    ]
    
    # Check if products table is empty before inserting
    c.execute('SELECT COUNT(*) FROM products')
    if c.fetchone()[0] == 0:
        c.executemany('INSERT INTO products (name, price, category, stock) VALUES (%s, %s, %s, %s)', sample_products)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

# Cart Functionality
def get_cart_count():
    if 'user_id' not in session:
        return 0
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT SUM(quantity) FROM cart WHERE user_id = %s', (session['user_id'],))
    count = c.fetchone()[0]
    conn.close()
    return count or 0

# System Authentication
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        preferences = request.form.get('preferences', '')

        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password, preferences) VALUES (%s, %s, %s)',
                      (username, password, preferences))
            conn.commit()
            return redirect(url_for('login'))
        except psycopg2.IntegrityError:
            conn.rollback()
            return 'Username already exists'
        finally:
            conn.close()
    return render_template('register.html', cart_count=0)
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('home'))
        return "Invalid Credentials"
    return render_template('login.html', cart_count=0)

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT p.id, p.name, p.price, p.category, c.quantity, p.stock
        FROM cart c
        JOIN products p ON c.product_id = p.id
        WHERE c.user_id = %s
    ''', (session['user_id'],))
    cart_items = c.fetchall()
    
    conn.close()
    
    cart_count = get_cart_count()
    return render_template('cart.html', cart_items=cart_items, cart_count=cart_count)

# Cart Tracker
@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        # Get cart items
        c.execute('''
            SELECT c.product_id, c.quantity, p.stock, p.name
            FROM cart c
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = %s
        ''', (session['user_id'],))
        cart_items = c.fetchall()
        
        if not cart_items:
            return jsonify({'success': False, 'message': 'Your cart is empty'}), 400
        
        # Check stock availability
        for item in cart_items:
            product_id, quantity, stock, name = item
            if quantity > stock:
                return jsonify({'success': False, 'message': f'Not enough stock for {name}'}), 400
        
        # Create orders and update stock
        timestamp = datetime.now().isoformat()
        for item in cart_items:
            product_id, quantity, _, _ = item
            c.execute('INSERT INTO orders (user_id, product_id, quantity, timestamp, status) VALUES (%s, %s, %s, %s, %s)',
                      (session['user_id'], product_id, quantity, timestamp, 'Pending'))
            c.execute('UPDATE products SET stock = stock - %s WHERE id = %s', (quantity, product_id))
        
        # Clear cart
        c.execute('DELETE FROM cart WHERE user_id = %s', (session['user_id'],))
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Checkout successful! Your order has been placed.'})
    
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({'success': False, 'message': 'An error occurred during checkout'}), 500
    finally:
        conn.close()

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
    
# E-commerce functionality
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    c = conn.cursor()
    
    # Verify user exists
    c.execute('SELECT id FROM users WHERE id = %s', (session['user_id'],))
    user = c.fetchone()
    
    if user is None:
        session.pop('user_id', None)
        conn.close()
        return redirect(url_for('login'))
    
    c.execute('SELECT * FROM products WHERE stock > 0')
    products = c.fetchall()
    recommendations = get_personalized_recommendations(session['user_id'])
    
    conn.close()
    
    cart_count = get_cart_count()
    return render_template('home.html', products=products, recommendations=recommendations, cart_count=cart_count)

@app.route('/buy/<int:product_id>', methods=['POST'])
def buy(product_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    quantity = int(request.form.get('quantity', 1))
    
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        # Check stock availability
        c.execute('SELECT stock, name FROM products WHERE id = %s', (product_id,))
        product = c.fetchone()
        if not product:
            return jsonify({'success': False, 'message': 'Product not found'}), 404
        
        if product[0] < quantity:
            return jsonify({'success': False, 'message': 'Not enough stock'}), 400
        
        # Check if item is already in cart
        c.execute('SELECT quantity FROM cart WHERE user_id = %s AND product_id = %s',
                  (session['user_id'], product_id))
        cart_item = c.fetchone()
        
        if cart_item:
            new_quantity = cart_item[0] + quantity
            c.execute('UPDATE cart SET quantity = %s WHERE user_id = %s AND product_id = %s',
                      (new_quantity, session['user_id'], product_id))
        else:
            c.execute('INSERT INTO cart (user_id, product_id, quantity) VALUES (%s, %s, %s)',
                      (session['user_id'], product_id, quantity))
        
        conn.commit()
        return jsonify({
            'success': True,
            'message': f'{product[1]} added to cart!',
            'product_id': product_id,
            'quantity': quantity
        })
    
    except psycopg2.Error as e:
        conn.rollback()
        return jsonify({'success': False, 'message': 'An error occurred'}), 500
    finally:
        conn.close()

# Personalized Recommendation System
def get_personalized_recommendations(user_id):
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get user preferences and purchase history
    c.execute('SELECT preferences FROM users WHERE id = %s', (user_id,))
    user_data = c.fetchone()
    
    preferences = user_data[0].split(',') if user_data and user_data[0] else []
    c.execute('SELECT product_id FROM orders WHERE user_id = %s', (user_id,))
    purchased_products = set(order[0] for order in c.fetchall())
    
    rec_products = []
    
    if preferences:
        placeholders = ','.join('%s' for _ in preferences)
        c.execute(
            f'SELECT * FROM products WHERE category IN ({placeholders}) AND stock > 0 AND id NOT IN %s',
            preferences + [tuple(purchased_products)] if purchased_products else preferences + [()]
        )
        rec_products = c.fetchall()
    
    # If less than 2 recommendations, add popular items
    if len(rec_products) < 2:
        c.execute('''
            SELECT p.*, COUNT(o.product_id) as order_count 
            FROM products p 
            LEFT JOIN orders o ON p.id = o.product_id 
            WHERE p.stock > 0 AND p.id NOT IN %s
            GROUP BY p.id, p.name, p.price, p.category, p.stock 
            ORDER BY order_count DESC 
            LIMIT 2
        ''', (tuple(purchased_products),) if purchased_products else ((),))
        rec_products.extend(c.fetchall())
    
    conn.close()
    return rec_products[:2]

# Run App
if __name__ == '__main__':
    init_db()
    # For production, use a WSGI server like Gunicorn
    # Example: gunicorn --workers 4 app:app
    # Do not use debug=True or ssl_context='adhoc' in production
    app.run()