import sqlite3
import psycopg2
import os
# Define BASE_DIR for consistent project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_db_connection():
    try:
        # Use environment variable for database path, default to project root
        db_path = os.environ.get('DATABASE_PATH', os.path.join(BASE_DIR, 'ecommerce.db'))
        
        # Ensure the directory exists (only for non-root paths)
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            print(f"Creating directory: {db_dir}")
            os.makedirs(db_dir, exist_ok=True)
        
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        print("Database connection successful")
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in get_db_connection: {e}")
        return None

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