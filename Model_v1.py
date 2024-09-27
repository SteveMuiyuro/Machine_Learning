import os
import json
import re
import requests
import spacy
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables from the .env file
load_dotenv()

# Access the API keys
place_key = os.getenv('PLACE_API_KEY')
search_key = os.getenv('SEARCH_API_KEY')
search_engine_id = os.getenv('SEARCH_ENGINE_ID')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")


# Extract product and city using NLP
def extract_product_and_city(prompt):
    doc = nlp(prompt)
    city = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), None)

    product_pattern = re.search(r'suppliers\s+([\w\s]+)(?:\s+' + re.escape(city) + ')?', prompt, re.IGNORECASE)
    product = product_pattern.group(1).strip() if product_pattern else None

    return product, city


# Fetch product price from Custom Search API
def fetch_product_price_from_custom_search(product_name, website_url):
    base_url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': search_key,
        'cx': search_engine_id,
        'q': f"{product_name} site:{website_url}",
        'num': 1
    }

    response = requests.get(base_url, params=params)

    if response.ok:
        items = response.json().get('items', [])
        if items:
            first_item = items[0]
            og_image = first_item.get('pagemap', {}).get('metatags', [{}])[0].get('og:image')
            product_info = {
                'title': first_item.get('title'),
                'link': first_item.get('link'),
                'image': og_image,
                'description': first_item.get('pagemap', {}).get('metatags', [{}])[0].get('og:description'),
                'price': scrape_product_price(first_item.get('link'))  # Updated function call
            }

            return product_info
    return None


# Scrape product price from the page
def scrape_product_price(page_url, query_params=None, retries=3, delay=2):
    if query_params is None:
        query_params = {}

    for attempt in range(retries):
        try:
            response = requests.get(page_url, params=query_params, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses

            soup = BeautifulSoup(response.text, 'html.parser')


            # Possible tags and classes to check for the price
            price_selectors = [
                {'tag': 'span', 'class': 'price'},
                {'tag': 'span', 'class': 'product-price'},
                {'tag': 'span', 'class': 'current-price'},
                {'tag': 'span', 'class': 'final-price'},
                {'tag': 'div', 'class': 'price'},
                {'tag': 'div', 'class': 'product-price'},
                {'tag': 'p', 'class': 'price'},
                {'tag': 'p', 'class': 'product-price'},
                {'tag': 'strong', 'class': 'price'},
                {'tag': 'strong', 'class': 'product-price'},
                {'tag': 'span', 'class': 'discounted-price'},  # Example of discounted price tag
            ]

            # Try to find the price using the specified selectors
            price = None
            for selector in price_selectors:
                price_tag = soup.find(selector['tag'], class_=selector['class'])
                if price_tag:
                    price = price_tag.text.strip()
                    break

            # If no price found, try to find it with regex in the entire text
            if not price:
                # Regex pattern to capture various currency formats (e.g., $10.99, KSh 500, etc.)
                price_pattern = re.compile(r'(\$|KSh|€|£)?\s*\d+(\.\d{1,2})?')
                price_matches = price_pattern.findall(soup.get_text())
                if price_matches:
                    # Pick the first match as the price
                    price = ''.join(price_matches[0]).strip()

            return price if price else "Price not found"

        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"Attempt {attempt + 1}/{retries} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return "Request failed"

    print("Max retries exceeded. Unable to fetch the product price.")
    return "Max retries exceeded"


# Get a list of suppliers from Google Places API
def get_suppliers_from_google(product, location):
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {'query': f"{product} suppliers in {location}", 'key': place_key}

    response = requests.get(base_url, params=params)
    return response.json().get('results', []) if response.ok else []


# Get supplier website by place_id
def get_supplier_website(supplier):
    place_id = supplier.get('place_id')
    if place_id:
        place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'fields': 'website',
            'key': place_key,
        }
        response = requests.get(place_details_url, params=params)

        return response.json().get('result', {}).get('website', 'N/A') if response.ok else 'N/A'
    return 'N/A'


# Calculate distances and scores in parallel
def calculate_distances_and_scores(suppliers, org_location, product_name):
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(calculate_distance_and_score, supplier, org_location, product_name): supplier
            for supplier in suppliers
        }
        return [future.result() for future in futures]


def calculate_distance_and_score(supplier, org_location, product_name):
    lat = supplier['geometry']['location']['lat']
    lng = supplier['geometry']['location']['lng']
    distance = geodesic((lat, lng), org_location).km
    rating = supplier.get('rating', 0)
    website = get_supplier_website(supplier)
    product_info = fetch_product_price_from_custom_search(product_name, website)

    return {
        'name': supplier.get('name'),
        'address': supplier.get('formatted_address'),
        'lat': lat,
        'lng': lng,
        'distance': distance,
        'rating': rating,
        'website': website,
        'product_info': product_info
    }


# Main functionality
organization_location = (-1.94623268784134, 30.067488122865363) ## Kigali Rwanda Coordinates

# Example usage
user_prompt = "suppliers of iphone 15 in Rwanda"
product, city = extract_product_and_city(user_prompt)

# Get suppliers from Google Places API
suppliers = get_suppliers_from_google(product, city)


# Calculate distances and scores for suppliers
supplier_data = calculate_distances_and_scores(suppliers, organization_location, product)

# Filter suppliers based on website availability and product info
filtered_suppliers = [
    supplier for supplier in supplier_data
    if supplier['website'] != 'N/A' and supplier['product_info'] is not None
]

# Sort suppliers by distance and rating (optional)
sorted_suppliers = sorted(filtered_suppliers, key=lambda x: (x['distance'], -x['rating']))

# print(json.dumps(sorted_suppliers, indent=4))
