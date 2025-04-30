import sqlite3
import os
import requests
import time

def get_db_connection():
    conn = sqlite3.connect('ecommerce.db')
    conn.row_factory = sqlite3.Row
    return conn

# List of 30 new products
new_products = [
    ('Laptop Stand', 39.99, 'Electronics', 80, None),
    ('Running Shoes', 59.99, 'Clothing', 120, None),
    ('Desk Lamp', 24.99, 'Home Appliances', 90, None),
    ('USB-C Cable', 9.99, 'Electronics', 200, None),
    ('Winter Jacket', 89.99, 'Clothing', 60, None),
    ('Electric Kettle', 34.99, 'Home Appliances', 70, None),
    ('Gaming Keyboard', 69.99, 'Electronics', 50, None),
    ('Sweatshirt', 29.99, 'Clothing', 150, None),
    ('Blender', 49.99, 'Home Appliances', 40, None),
    ('Portable Charger', 19.99, 'Electronics', 100, None),
    ('Jeans', 39.99, 'Clothing', 110, None),
    ('Air Fryer', 79.99, 'Home Appliances', 30, None),
    ('Smartwatch', 129.99, 'Electronics', 25, None),
    ('Baseball Cap', 14.99, 'Clothing', 180, None),
    ('Toaster', 29.99, 'Home Appliances', 60, None),
    ('Wireless Earbuds', 39.99, 'Electronics', 90, None),
    ('Hoodie', 34.99, 'Clothing', 130, None),
    ('Microwave Oven', 99.99, 'Home Appliances', 20, None),
    ('Tablet', 299.99, 'Electronics', 15, None),
    ('Sneakers', 49.99, 'Clothing', 100, None),
    ('Vacuum Cleaner', 149.99, 'Home Appliances', 10, None),
    ('External Hard Drive', 79.99, 'Electronics', 40, None),
    ('Scarf', 12.99, 'Clothing', 200, None),
    ('Electric Fan', 44.99, 'Home Appliances', 50, None),
    ('Webcam', 59.99, 'Electronics', 30, None),
    ('Socks (Pack of 5)', 9.99, 'Clothing', 300, None),
    ('Food Processor', 69.99, 'Home Appliances', 25, None),
    ('Fitness Tracker', 49.99, 'Electronics', 35, None),
    ('Trench Coat', 99.99, 'Clothing', 40, None),
    ('Humidifier', 39.99, 'Home Appliances', 45, None),
]

# Update scrape_images.py to handle new products
UNSPLASH_ACCESS_KEY = 'vAptsvjq85WDg2kL5mNQWoRDoQ6HAGOdQGxSRWK4i_I'
image_dir = 'static/images/'
os.makedirs(image_dir, exist_ok=True)

def fetch_unsplash_image(product_name):
    try:
        search_url = "https://api.unsplash.com/search/photos"
        headers = {'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'}
        params = {'query': product_name, 'per_page': 1}
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if not data['results']:
            print(f"No image found on Unsplash for {product_name}")
            return None
        img_url = data['results'][0]['urls']['regular']
        img_filename = product_name.lower().replace(' ', '_') + '.jpg'
        img_path = os.path.join(image_dir, img_filename)
        img_response = requests.get(img_url, stream=True)
        img_response.raise_for_status()
        with open(img_path, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded image for {product_name} to {img_path}")
        return f"/static/images/{img_filename}"
    except Exception as e:
        print(f"Error fetching image for {product_name}: {e}")
        return None

# Fetch images for new products and update the product list
for product in new_products:
    image_path = fetch_unsplash_image(product[0])
    if image_path:
        product_list = list(product)
        product_list[4] = image_path
        new_products[new_products.index(product)] = tuple(product_list)
    time.sleep(1)  # Respect Unsplash API rate limits

# Insert products into the database
try:
    conn = get_db_connection()
    c = conn.cursor()
    c.executemany('INSERT INTO products (name, price, category, stock, image) VALUES (?, ?, ?, ?, ?)', new_products)
    conn.commit()
    print(f"Successfully added {len(new_products)} products to the database.")
except sqlite3.Error as e:
    print(f"Error adding products to database: {e}")
finally:
    conn.close()