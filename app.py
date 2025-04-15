import os
import sqlite3
import psycopg2
from flask import Flask, session, render_template, request, redirect, url_for, jsonify
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', '5507e4842f2c53c15f4a3bbd1e004e6ef59eb7007920c29d1c2b1bc133d90336')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set. Please set it in environment variables.")

# Database Connection (SQLite or PostgreSQL)
def get_db_connection():
    try:
        if os.environ.get('DATABASE_URL'):
            # PostgreSQL connection
            conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        else:
            # SQLite connection
            conn = sqlite3.connect('ecommerce.db')
            conn.row_factory = sqlite3.Row  # For dictionary-like access
        return conn
    except (psycopg2.Error, sqlite3.Error) as e:
        print(f"Database connection error: {e}")
        raise

# Initialize Database
def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                preferences TEXT
            )
        ''')
        
        # Products table
        c.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT,
                stock INTEGER NOT NULL
            )
        ''')
        
        # Orders table
        c.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                timestamp TEXT,
                status TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            )
        ''')
        
        # Cart table
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
        
        # Check if products table is empty
        c.execute('SELECT COUNT(*) FROM products')
        if c.fetchone()[0] == 0:
            c.executemany('INSERT INTO products (name, price, category, stock) VALUES (?, ?, ?, ?)', sample_products)
        
        conn.commit()
    except (psycopg2.Error, sqlite3.Error) as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()
    print("Database initialized successfully")

# Cart Count Helper
def get_cart_count():
    if 'user_id' not in session:
        return 0
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT SUM(quantity) FROM cart WHERE user_id = ?', (session['user_id'],))
        count = c.fetchone()[0]
        conn.close()
        return count or 0
    except (psycopg2.Error, sqlite3.Error):
        return 0

# System Authentication
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        preferences = request.form.get('preferences', '')
        if not username or not password:
            return render_template('register.html', error='Username and password are required', cart_count=0)
        
        hashed_password = generate_password_hash(password)
        try:
            conn = get_db_connection()
            c = conn.cursor()
            try:
                c.execute('INSERT INTO users (username, password, preferences) VALUES (?, ?, ?)',
                          (username, hashed_password, preferences))
                conn.commit()
                return redirect(url_for('login'))
            except (psycopg2.IntegrityError, sqlite3.IntegrityError):
                conn.rollback()
                return render_template('register.html', error='Username already exists', cart_count=0)
        except (psycopg2.Error, sqlite3.Error) as e:
            print(f"Register error: {e}")
            return render_template('register.html', error='Registration failed, try again', cart_count=0)
        finally:
            conn.close()
    return render_template('register.html', cart_count=0)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('login.html', error='Username and password are required', cart_count=0)
        
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            conn.close()
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session.permanent = True  # Persist for mobile
                return redirect(url_for('home'))
            return render_template('login.html', error='Invalid credentials', cart_count=0)
        except (psycopg2.Error, sqlite3.Error) as e:
            print(f"Login error: {e}")
            return render_template('login.html', error='Login failed, try again', cart_count=0)
    return render_template('login.html', cart_count=0)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# E-commerce Routes
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Verify user exists
        c.execute('SELECT id FROM users WHERE id = ?', (session['user_id'],))
        if not c.fetchone():
            session.clear()
            conn.close()
            return redirect(url_for('login'))
        
        c.execute('SELECT * FROM products WHERE stock > 0')
        products = c.fetchall()
        recommendations = get_personalized_recommendations(session['user_id'])
        
        conn.close()
        cart_count = get_cart_count()
        
        response = render_template('home.html', products=products, recommendations=recommendations, cart_count=cart_count)
        response = app.make_response(response)
        response.headers['Cache-Control'] = 'public, max-age=60'
        return response
    except (psycopg2.Error, sqlite3.Error) as e:
        print(f"Home error: {e}")
        return render_template('error.html', error='Failed to load products, try again', cart_count=0)

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        c.execute('''
            SELECT p.id, p.name, p.price, p.category, c.quantity, p.stock
            FROM cart c
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = ?
        ''', (session['user_id'],))
        cart_items = c.fetchall()
        
        conn.close()
        cart_count = get_cart_count()
        return render_template('cart.html', cart_items=cart_items, cart_count=cart_count)
    except (psycopg2.Error, sqlite3.Error) as e:
        print(f"Cart error: {e}")
        return render_template('cart.html', cart_items=[], error='Failed to load cart', cart_count=0)

@app.route('/buy/<int:product_id>', methods=['POST'])
def buy(product_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    quantity = int(request.form.get('quantity', 1))
    if quantity < 1:
        return jsonify({'success': False, 'message': 'Invalid quantity'}), 400
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Check stock
        c.execute('SELECT stock, name FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()
        if not product:
            return jsonify({'success': False, 'message': 'Product not found'}), 404
        if product['stock'] < quantity:
            return jsonify({'success': False, 'message': f'Not enough stock for {product["name"]}'}), 400
        
        # Update or insert cart
        c.execute('SELECT quantity FROM cart WHERE user_id = ? AND product_id = ?',
                  (session['user_id'], product_id))
        cart_item = c.fetchone()
        
        if cart_item:
            new_quantity = cart_item['quantity'] + quantity
            c.execute('UPDATE cart SET quantity = ? WHERE user_id = ? AND product_id = ?',
                      (new_quantity, session['user_id'], product_id))
        else:
            c.execute('INSERT INTO cart (user_id, product_id, quantity) VALUES (?, ?, ?)',
                      (session['user_id'], product_id, quantity))
        
        conn.commit()
        return jsonify({
            'success': True,
            'message': f'{product["name"]} added to cart!',
            'product_id': product_id,
            'quantity': quantity
        })
    except (psycopg2.Error, sqlite3.Error) as e:
        conn.rollback()
        print(f"Buy error: {e}")
        return jsonify({'success': False, 'message': 'Failed to add to cart'}), 500
    finally:
        conn.close()

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get cart items
        c.execute('''
            SELECT c.product_id, c.quantity, p.stock, p.name
            FROM cart c
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = ?
        ''', (session['user_id'],))
        cart_items = c.fetchall()
        
        if not cart_items:
            return jsonify({'success': False, 'message': 'Your cart is empty'}), 400
        
        # Check stock
        for item in cart_items:
            product_id, quantity, stock, name = item['product_id'], item['quantity'], item['stock'], item['name']
            if quantity > stock:
                return jsonify({'success': False, 'message': f'Not enough stock for {name}'}), 400
        
        # Create orders and update stock
        timestamp = datetime.now().isoformat()
        for item in cart_items:
            product_id, quantity = item['product_id'], item['quantity']
            c.execute('INSERT INTO orders (user_id, product_id, quantity, timestamp, status) VALUES (?, ?, ?, ?, ?)',
                      (session['user_id'], product_id, quantity, timestamp, 'Pending'))
            c.execute('UPDATE products SET stock = stock - ? WHERE id = ?', (quantity, product_id))
        
        # Clear cart
        c.execute('DELETE FROM cart WHERE user_id = ?', (session['user_id'],))
        
        conn.commit()
        return jsonify({'success': True, 'message': 'Checkout successful! Your order has been placed.'})
    except (psycopg2.Error, sqlite3.Error) as e:
        conn.rollback()
        print(f"Checkout error: {e}")
        return jsonify({'success': False, 'message': 'Checkout failed'}), 500
    finally:
        conn.close()

# Personalized Recommendations
def get_personalized_recommendations(user_id):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get user preferences and purchase history
        c.execute('SELECT preferences FROM users WHERE id = ?', (user_id,))
        user_data = c.fetchone()
        preferences = user_data['preferences'].split(',') if user_data and user_data['preferences'] else []
        
        c.execute('SELECT product_id FROM orders WHERE user_id = ?', (user_id,))
        purchased_products = set(order['product_id'] for order in c.fetchall())
        
        rec_products = []
        
        if preferences:
            placeholders = ','.join('?' for _ in preferences)
            c.execute(
                f'SELECT * FROM products WHERE category IN ({placeholders}) AND stock > 0 AND id NOT IN ({",".join("?" for _ in purchased_products)})',
                preferences + list(purchased_products) if purchased_products else preferences
            )
            rec_products = c.fetchall()
        
        # Add popular items if needed
        if len(rec_products) < 2:
            c.execute('''
                SELECT p.*, COUNT(o.product_id) as order_count 
                FROM products p 
                LEFT JOIN orders o ON p.id = o.product_id 
                WHERE p.stock > 0 AND p.id NOT IN ({})
                GROUP BY p.id 
                ORDER BY order_count DESC 
                LIMIT 2
            '''.format(','.join('?' for _ in purchased_products)), list(purchased_products) if purchased_products else [])
            rec_products.extend(c.fetchall())
        
        conn.close()
        return rec_products[:2]
    except (psycopg2.Error, sqlite3.Error) as e:
        print(f"Recommendations error: {e}")
        return []

# Error Page
@app.route('/error')
def error():
    return render_template('error.html', error='An unexpected error occurred', cart_count=get_cart_count())

# Run App
if __name__ == '__main__':
    try:
        init_db()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Startup error: {e}")