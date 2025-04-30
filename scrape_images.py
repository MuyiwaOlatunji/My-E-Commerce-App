import requests
import os
import time

# Update the products list to include initial products
products = [
    'Wireless Mouse',
    'Graphic T-shirt',
    'Bluetooth Headphones',
    'Coffee Maker',
    'Smartphone',
    'Laptop Stand',
    'Running Shoes',
    'Desk Lamp',
    'USB-C Cable',
    'Winter Jacket',
    'Electric Kettle',
    'Gaming Keyboard',
    'Sweatshirt',
    'Blender',
    'Portable Charger',
    'Jeans',
    'Air Fryer',
    'Smartwatch',
    'Baseball Cap',
    'Toaster',
    'Wireless Earbuds',
    'Hoodie',
    'Microwave Oven',
    'Tablet',
    'Sneakers',
    'Vacuum Cleaner',
    'External Hard Drive',
    'Scarf',
    'Electric Fan',
    'Webcam',
    'Socks (Pack of 5)',
    'Food Processor',
    'Fitness Tracker',
    'Trench Coat',
    'Humidifier',
]

image_dir = 'static/images/'
os.makedirs(image_dir, exist_ok=True)

UNSPLASH_ACCESS_KEY = 'vAptsvjq85WDg2kL5mNQWoRDoQ6HAGOdQGxSRWK4i_I'  # Ensure this is your key

def fetch_unsplash_image(product_name):
    try:
        search_url = f"https://api.unsplash.com/search/photos"
        headers = {
            'Authorization': f'Client-ID {UNSPLASH_ACCESS_KEY}'
        }
        params = {
            'query': product_name,
            'per_page': 1
        }
        
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data['results']:
            print(f"No image found on Unsplash for {product_name}")
            return None
        
        img_url = data['results'][0]['urls']['regular']
        print(f"Image URL for {product_name}: {img_url}")  # Debug
        
        img_filename = product_name.lower().replace(' ', '_').replace('_(pack_of_5)', '') + '.jpg'
        img_path = os.path.join(image_dir, img_filename)
        
        # Download the image using requests
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

for product in products:
    image_path = fetch_unsplash_image(product)
    if image_path:
        print(f"Image path for {product}: {image_path}")
    else:
        print(f"Failed to fetch image for {product}")
        img_filename = product.lower().replace(' ', '_').replace('_(pack_of_5)', '') + '.jpg'
        image_path = f"/static/images/{img_filename}"
        print(f"Using placeholder path: {image_path}")
    time.sleep(1)