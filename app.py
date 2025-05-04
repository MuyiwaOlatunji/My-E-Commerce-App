import os
import sqlite3
from flask import Flask, session, render_template, request, redirect, url_for, jsonify, make_response
from flask_wtf.csrf import CSRFProtect
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import requests
from dotenv import load_dotenv
import uuid
from forms import LoginForm, RegisterForm, CheckoutForm, PaymentForm
import signal
import sys

# Handle PyInstaller runtime paths
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    os.chdir(BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), template_folder=os.path.join(BASE_DIR, 'templates'))
app.secret_key = os.environ.get('SECRET_KEY', '5507e4842f2c53c15f4a3bbd1e004e6ef59eb7007920c29d1c2b1bc133d90336')
if not app.secret_key:
    raise ValueError("No SECRET_KEY set.")
csrf = CSRFProtect(app)

# Initialize caching
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)

# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {str(e)}")
    import traceback
    traceback.print_exc()
    return render_template('error.html', error='An unexpected error occurred: ' + str(e), cart_count=get_cart_count()), 500

# Database Connection
def get_db_connection():
    try:
        db_path = os.environ.get('DATABASE_PATH', os.path.join(BASE_DIR, 'ecommerce.db'))
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        print("Database connection successful")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# Initialize Database
def init_db():
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            raise Exception("Failed to connect to the database")
        c = conn.cursor()
        
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
        
        c.execute('SELECT COUNT(*) FROM products')
        if c.fetchone()[0] == 0:
            c.executemany('INSERT INTO products (name, price, category, stock, image) VALUES (?, ?, ?, ?, ?)', sample_products)
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
    finally:
        if conn is not None:
            conn.close()

# Cart Count Helper
def get_cart_count():
    if 'user_id' not in session:
        return 0
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return 0
        c = conn.cursor()
        c.execute('SELECT SUM(quantity) FROM cart WHERE user_id = ?', (session['user_id'],))
        count = c.fetchone()[0]
        return count or 0
    except sqlite3.Error:
        return 0
    finally:
        if conn is not None:
            conn.close()

# PyTorch NCF Recommendation Model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[128, 64, 32]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.ModuleList()
        input_size = embedding_dim * 2
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

# Routes
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return render_template('error.html', error='Database connection failed', cart_count=0)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE id = ?', (session['user_id'],))
        if not c.fetchone():
            session.clear()
            return redirect(url_for('login'))
        c.execute('SELECT * FROM products WHERE stock > 0')
        products = c.fetchall()
        print("Products fetched for home page:", [(p['name'], p['image']) for p in products])
        cart_count = get_cart_count()
        rec_response = recommendations()
        if isinstance(rec_response, tuple):
            rec_data = rec_response[0].get_json()
            status = rec_response[1]
            if status != 200:
                print(f"Recommendations failed with status {status}: {rec_data}")
                rec_data = []
        else:
            rec_data = rec_response.get_json()
        if not isinstance(rec_data, list):
            print(f"Invalid recommendations format: {rec_data}")
            rec_data = []
        elif rec_data and isinstance(rec_data[0], dict) and 'error' in rec_data[0]:
            print(f"Recommendations error: {rec_data[0]['error']}")
            rec_data = []
        else:
            print("Recommendations with images:", [(r['name'], r.get('image', '')) for r in rec_data])
        response = render_template('home.html', products=products, recommendations=rec_data, cart_count=cart_count)
        response = make_response(response)
        response.headers['Cache-Control'] = 'public, max-age=60'
        return response
    except sqlite3.Error as e:
        print(f"Home error: {e}")
        return render_template('error.html', error='Failed to load products', cart_count=0)
    finally:
        if conn is not None:
            conn.close()

@app.route('/search')
def search():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    query = request.args.get('q', '')
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return render_template('error.html', error='Database connection failed', cart_count=0)
        c = conn.cursor()
        c.execute('SELECT * FROM products WHERE name LIKE ? AND stock > 0', (f'%{query}%',))
        products = c.fetchall()
        cart_count = get_cart_count()
        return render_template('home.html', products=products, cart_count=cart_count)
    except sqlite3.Error as e:
        print(f"Search error: {e}")
        return render_template('error.html', error='Search failed', cart_count=0)
    finally:
        if conn is not None:
            conn.close()

@app.route('/cart')
def cart():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    print(f"Fetching cart for user_id: {session['user_id']}")
    form = CheckoutForm()
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return render_template('error.html', error='Database connection failed', cart_count=0)
        c = conn.cursor()
        
        c.execute('SELECT * FROM cart WHERE user_id = ?', (session['user_id'],))
        cart_entries = c.fetchall()
        print(f"Raw cart entries: {cart_entries}")
        
        if not cart_entries:
            print("No cart entries found for this user.")
            cart_count = get_cart_count()
            return render_template('cart.html', cart_items=[], total=0, cart_count=cart_count, form=form, is_cart_empty=True)

        c.execute('''
            SELECT p.id, p.name, p.price, p.category, c.quantity, p.stock, p.image
            FROM cart c
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = ?
        ''', (session['user_id'],))
        cart_items = c.fetchall()
        print(f"Cart items after join: {cart_items}")
        
        total = sum(item['price'] * item['quantity'] for item in cart_items)
        cart_count = get_cart_count()
        return render_template('cart.html', cart_items=cart_items, total=total, cart_count=cart_count, form=form, is_cart_empty=False)
    
    except sqlite3.Error as e:
        print(f"Cart error: {e}")
        cart_count = get_cart_count()
        return render_template('cart.html', cart_items=[], error=f'Failed to load cart: {str(e)}', cart_count=cart_count, form=form, is_cart_empty=True)
    finally:
        if conn is not None:
            conn.close()

@app.route('/buy/<int:product_id>', methods=['POST'])
@csrf.exempt
def buy(product_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    quantity = int(request.form.get('quantity', 1))
    if quantity < 1:
        return jsonify({'success': False, 'message': 'Invalid quantity'}), 400
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        c = conn.cursor()
        c.execute('SELECT stock, name FROM products WHERE id = ?', (product_id,))
        product = c.fetchone()
        if not product:
            return jsonify({'success': False, 'message': 'Product not found'}), 404
        if product['stock'] < quantity:
            return jsonify({'success': False, 'message': f'Not enough stock for {product["name"]}'}), 400
        print(f"Adding to cart for user_id: {session['user_id']}")
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
        c.execute('INSERT INTO interactions (user_id, product_id, action, timestamp) VALUES (?, ?, ?, ?)',
                  (session['user_id'], product_id, 'view', datetime.now().isoformat()))
        conn.commit()
        c.execute('SELECT * FROM cart WHERE user_id = ?', (session['user_id'],))
        cart_contents = c.fetchall()
        print(f"Cart contents after adding: {cart_contents}")
        return jsonify({
            'success': True,
            'message': f'{product["name"]} added to cart!',
            'product_id': product_id,
            'quantity': quantity
        })
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        print(f"Buy error: {e}")
        return jsonify({'success': False, 'message': 'Failed to add to cart'}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login'}), 401
    print(f"Checkout request received for user_id: {session['user_id']}")
    print(f"Form data: {request.form}")
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'success': False, 'message': 'Database connection failed'}), 500
        c = conn.cursor()
        
        c.execute('SELECT id FROM users WHERE id = ?', (session['user_id'],))
        if not c.fetchone():
            session.clear()
            return jsonify({'success': False, 'message': 'User not found'}), 401
        
        c.execute('''
            SELECT c.id, p.id as product_id, p.name, p.price, c.quantity, p.stock
            FROM cart c
            JOIN products p ON c.product_id = p.id
            WHERE c.user_id = ?
        ''', (session['user_id'],))
        cart_items = c.fetchall()
        
        print(f"Cart items: {[(item['name'], item['quantity'], item['stock']) for item in cart_items]}")
        
        if not cart_items:
            return jsonify({'success': False, 'message': 'Your cart is empty'}), 400
        
        for item in cart_items:
            if item['stock'] < item['quantity']:
                return jsonify({'success': False, 'message': f'Not enough stock for {item["name"]}'}), 400
        
        total_price = sum(item['price'] * item['quantity'] for item in cart_items)
        timestamp = datetime.now().isoformat()
        
        c.execute('INSERT INTO orders (user_id, total_price, status, timestamp) VALUES (?, ?, ?, ?)',
                  (session['user_id'], total_price, 'pending', timestamp))
        order_id = c.lastrowid
        
        for item in cart_items:
            c.execute('INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (?, ?, ?, ?)',
                      (order_id, item['product_id'], item['quantity'], item['price']))
            c.execute('UPDATE products SET stock = stock - ? WHERE id = ?', (item['quantity'], item['product_id']))
        
        c.execute('DELETE FROM cart WHERE user_id = ?', (session['user_id'],))
        
        conn.commit()
        print(f"Order created successfully: order_id={order_id}, total_price={total_price}")
        return jsonify({
            'success': True,
            'message': 'Checkout successful',
            'redirect': url_for('order_confirmation', order_id=order_id, _external=True)
        }), 200
    
    except sqlite3.Error as e:
        print(f"Checkout error: {e}")
        if conn:
            conn.rollback()
        return jsonify({'success': False, 'message': f'Failed to process checkout: {str(e)}'}), 500
    except Exception as e:
        print(f"Unexpected checkout error: {e}")
        return jsonify({'success': False, 'message': 'Unexpected error during checkout'}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/order_confirmation/<int:order_id>')
def order_confirmation(order_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return render_template('error.html', error='Database connection failed', cart_count=0)
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE id = ?', (session['user_id'],))
        if not c.fetchone():
            session.clear()
            return redirect(url_for('login'))
        c.execute('SELECT * FROM orders WHERE id = ? AND user_id = ?', (order_id, session['user_id']))
        order = c.fetchone()
        if not order:
            return render_template('error.html', error='Order not found', cart_count=get_cart_count()), 404
        c.execute('''
            SELECT oi.quantity, oi.price, p.name, p.image
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = ?
        ''', (order_id,))
        order_items = c.fetchall()
        cart_count = get_cart_count()
        return render_template('order_confirmation.html', order=order, order_items=order_items, cart_count=cart_count)
    except sqlite3.Error as e:
        print(f"Order confirmation error: {e}")
        return render_template('error.html', error='Failed to load order confirmation', cart_count=0), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    order_id = request.args.get('order_id', type=int)
    if not order_id:
        return render_template('error.html', error='Order ID is required', cart_count=get_cart_count()), 400
    form = PaymentForm()
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return render_template('error.html', error='Database connection failed', cart_count=0), 500
        c = conn.cursor()
        
        c.execute('SELECT * FROM orders WHERE id = ? AND user_id = ?', (order_id, session['user_id']))
        order = c.fetchone()
        if not order:
            return render_template('error.html', error='Order not found', cart_count=get_cart_count()), 404
        
        c.execute('''
            SELECT oi.quantity, oi.price, p.name, p.image
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            WHERE oi.order_id = ?
        ''', (order_id,))
        order_items = c.fetchall()
        
        if request.method == 'POST' and form.validate_on_submit():
            crypto = form.crypto.data
            try:
                print(f"NOWPayments API Key: {os.getenv('NOWPAYMENTS_API_KEY')}")
                total = order['total_price']
                payment_id = str(uuid.uuid4())
                response = requests.post(
                    'https://api.nowpayments.io/v1/payment',
                    headers={'x-api-key': os.getenv('NOWPAYMENTS_API_KEY')},
                    json={
                        'price_amount': total,
                        'price_currency': 'USD',
                        'pay_currency': crypto,
                        'order_id': payment_id,
                        'success_url': url_for('home', _external=True),
                        'cancel_url': url_for('payment', order_id=order_id, _external=True),
                    },
                ).json()
                
                if 'pay_address' in response:
                    c.execute('UPDATE orders SET payment_id = ? WHERE id = ?', (payment_id, order_id))
                    conn.commit()
                    return render_template('payment.html', address=response['pay_address'], crypto=crypto, payment_id=payment_id, order=order, order_items=order_items, cart_count=get_cart_count())
                else:
                    return render_template('error.html', error='Payment initiation failed: ' + str(response.get('message', 'Unknown error')), cart_count=get_cart_count()), 500
            except requests.RequestException as e:
                print(f"Payment error: {e}")
                return render_template('error.html', error='Payment processing failed', cart_count=get_cart_count()), 500
        
        return render_template('payment.html', form=form, order=order, order_items=order_items, cart_count=get_cart_count())
    
    except sqlite3.Error as e:
        print(f"Payment error: {e}")
        return render_template('error.html', error='Failed to load payment page', cart_count=get_cart_count()), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/payment/status/<payment_id>')
def payment_status(payment_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Please login'}), 401
    conn = None
    try:
        response = requests.get(
            f'https://api.nowpayments.io/v1/payment/{payment_id}',
            headers={'x-api-key': os.getenv('NOWPAYMENTS_API_KEY')}
        ).json()
        if response.get('payment_status') == 'confirmed':
            conn = get_db_connection()
            if conn is None:
                return jsonify({'status': 'error'}), 500
            c = conn.cursor()
            c.execute('UPDATE orders SET status = ? WHERE payment_id = ?', ('Confirmed', payment_id))
            conn.commit()
            return jsonify({'status': 'success', 'redirect': url_for('home', _external=True)})
        return jsonify({'status': 'pending'})
    except (sqlite3.Error, requests.RequestException) as e:
        print(f"Payment status error: {e}")
        return jsonify({'status': 'error'}), 500
    finally:
        if conn is not None:
            conn.close()

@app.route('/api/recommendations')
@cache.memoize(3600)
def recommendations():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login'}), 401
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        c = conn.cursor()
        
        c.execute('SELECT preferences FROM users WHERE id = ?', (session['user_id'],))
        user = c.fetchone()
        user_preference = user['preferences'] if user and user['preferences'] else None
        print(f"User ID: {session['user_id']}, Preference: {user_preference}")
        
        c.execute('SELECT COUNT(DISTINCT id) FROM users')
        num_users = max(c.fetchone()[0], 1)
        c.execute('SELECT COUNT(*) FROM products')
        num_items = max(c.fetchone()[0], 1)
        
        model = NCF(num_users, num_items, embedding_dim=64, layers=[128, 64, 32])
        model_path = os.path.join(BASE_DIR, 'model.pth')
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Model loading error: {e}")
            c.execute('''
                SELECT p.id, p.name, p.price, p.category, p.image
                FROM products p
                LEFT JOIN interactions i ON p.id = i.product_id
                WHERE p.stock > 0
                GROUP BY p.id
                ORDER BY COUNT(i.product_id) DESC
                LIMIT 5
            ''')
            products = c.fetchall()
            return jsonify([{
                'id': p['id'],
                'name': p['name'],
                'price': p['price'],
                'category': p['category'],
                'image': p['image'] or ''
            } for p in products])

        c.execute('SELECT id, category FROM products WHERE stock > 0')
        product_info = c.fetchall()
        product_ids = [row['id'] for row in product_info]
        product_categories = {row['id']: row['category'] for row in product_info}
        
        user_id = session['user_id']
        user_id_adjusted = user_id - 1
        
        if user_id_adjusted >= num_users:
            print(f"User ID {user_id} not in model, using fallback")
            c.execute('''
                SELECT p.id, p.name, p.price, p.category, p.image
                FROM products p
                LEFT JOIN interactions i ON p.id = i.product_id
                WHERE p.stock > 0
                GROUP BY p.id
                ORDER BY COUNT(i.product_id) DESC
                LIMIT 5
            ''')
            products = c.fetchall()
            return jsonify([{
                'id': p['id'],
                'name': p['name'],
                'price': p['price'],
                'category': p['category'],
                'image': p['image'] or ''
            } for p in products])

        product_ids_adjusted = [pid - 1 for pid in product_ids]
        user_ids = torch.tensor([user_id_adjusted] * len(product_ids_adjusted), dtype=torch.long)
        product_ids_tensor = torch.tensor(product_ids_adjusted, dtype=torch.long)
        
        with torch.no_grad():
            predictions = model(user_ids, product_ids_tensor).numpy().flatten()
        
        adjusted_scores = []
        for pred_score, pid_adjusted in zip(predictions, product_ids_adjusted):
            pid = pid_adjusted + 1
            category = product_categories.get(pid, '')
            if user_preference and user_preference.lower() == category.lower():
                adjusted_score = pred_score + 0.2
            else:
                adjusted_score = pred_score
            adjusted_scores.append((adjusted_score, pid_adjusted))
        
        top_ids = sorted(adjusted_scores, reverse=True)[:5]
        top_ids = [pid for _, pid in top_ids]
        top_ids = [pid + 1 for pid in top_ids]
        
        c.execute('SELECT id, name, price, category, image FROM products WHERE id IN ({}) AND stock > 0'.format(','.join('?' * len(top_ids))), top_ids)
        products = c.fetchall()
        
        if not products:
            print("No valid recommendations, using fallback")
            c.execute('''
                SELECT p.id, p.name, p.price, p.category, p.image
                FROM products p
                LEFT JOIN interactions i ON p.id = i.product_id
                WHERE p.stock > 0
                GROUP BY p.id
                ORDER BY COUNT(i.product_id) DESC
                LIMIT 5
            ''')
            products = c.fetchall()
        
        return jsonify([{
            'id': p['id'],
            'name': p['name'],
            'price': p['price'],
            'category': p['category'],
            'image': p['image'] or ''
        } for p in products])
    
    except sqlite3.Error as e:
        print(f"Recommendations error: {e}")
        c.execute('SELECT id, name, price, category, image FROM products WHERE stock > 0 ORDER BY RANDOM() LIMIT 5')
        products = c.fetchall()
        return jsonify([{
            'id': p['id'],
            'name': p['name'],
            'price': p['price'],
            'category': p['category'],
            'image': p['image'] or ''
        } for p in products])
    finally:
        if conn is not None:
            conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    print(f"Received request: method={request.method}, form data={form.data}")
    if request.method == 'POST':
        print("POST request received")
        if form.validate_on_submit():
            print("Form validation successful")
            username = form.username.data
            password = form.password.data
            preferences = form.preferences.data
            hashed_password = generate_password_hash(password)
            print(f"Registering user: username={username}, preferences={preferences}")
            conn = None
            try:
                conn = get_db_connection()
                if conn is None:
                    print("Database connection failed")
                    return render_template('register.html', form=form, error='Database connection failed', cart_count=0)
                c = conn.cursor()
                try:
                    c.execute('INSERT INTO users (username, password, preferences) VALUES (?, ?, ?)',
                              (username, hashed_password, preferences))
                    user_id = c.lastrowid
                    print(f"User inserted with ID: {user_id}")
                    c.execute('SELECT id FROM products WHERE stock > 0 ORDER BY RANDOM() LIMIT 2')
                    product_ids = [row['id'] for row in c.fetchall()]
                    print(f"Selected product IDs for interactions: {product_ids}")
                    interactions = [
                        (user_id, pid, 'view', datetime.now().isoformat())
                        for pid in product_ids
                    ]
                    c.executemany('INSERT INTO interactions (user_id, product_id, action, timestamp) VALUES (?, ?, ?, ?)', interactions)
                    print(f"Inserted {len(interactions)} interactions for user ID {user_id}")
                    conn.commit()
                    print("Transaction committed successfully")
                    return redirect(url_for('login'))
                except sqlite3.IntegrityError as e:
                    conn.rollback()
                    print(f"IntegrityError: {e}")
                    return render_template('register.html', form=form, error='Username already exists', cart_count=0)
            except sqlite3.Error as e:
                print(f"Register error: {e}")
                return render_template('register.html', form=form, error='Registration failed', cart_count=0)
            finally:
                if conn is not None:
                    conn.close()
        else:
            print("Form validation failed")
            print(f"Form errors: {form.errors}")
    return render_template('register.html', form=form, cart_count=0)

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        print("Initializing LoginForm...")
        form = LoginForm()
        print("LoginForm initialized successfully")
        if request.method == 'POST' and form.validate_on_submit():
            username = form.username.data
            password = form.password.data
            conn = None
            try:
                conn = get_db_connection()
                if conn is None:
                    return render_template('login.html', form=form, error='Database connection failed', cart_count=0)
                c = conn.cursor()
                c.execute('SELECT * FROM users WHERE username = ?', (username,))
                user = c.fetchone()
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    session.permanent = True
                    print(f"User logged in with user_id: {session['user_id']}")
                    return redirect(url_for('home'))
                return render_template('login.html', form=form, error='Invalid credentials', cart_count=0)
            except sqlite3.Error as e:
                print(f"Login error: {e}")
                return render_template('login.html', form=form, error='Login failed', cart_count=0)
            finally:
                if conn is not None:
                    conn.close()
        return render_template('login.html', form=form, cart_count=0)
    except Exception as e:
        print(f"Error in /login route: {e}")
        raise e

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/error')
def error():
    return render_template('error.html', error='An unexpected error occurred', cart_count=get_cart_count())

# Graceful shutdown handler
def shutdown_server(signum, frame):
    print("Shutting down server...")
    sys.exit(0)

if __name__ == '__main__':
    init_db()
    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)
    port = int(os.getenv('Port', 10000))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)