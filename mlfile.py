import pandas as pd
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import requests
import spacy
from dotenv import load_dotenv
import os
import json
import re
from geopy.distance import geodesic

# Load environment variables from the .env file
load_dotenv()


# Access the API key
key = os.getenv('API_KEY')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Custom function to extract product and location
def extract_product_and_city(prompt):
    doc = nlp(prompt)

    # Extract city using NER (Location entity)
    city = None
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE (Geopolitical Entity) typically identifies cities, countries, etc.
            city = ent.text
            break

    # Extract product using regex, considering possible formats in the prompt
    # This regex will try to capture anything after "suppliers" but before the city or end of the string
    product_pattern = None
    if city:
        product_pattern = re.search(r'suppliers\s+([\w\s]+)\s+' + re.escape(city), prompt, re.IGNORECASE)
    else:
        product_pattern = re.search(r'suppliers\s+([\w\s]+)', prompt, re.IGNORECASE)

    product = product_pattern.group(1).strip() if product_pattern else None

    return product, city

# Google Places API Key
api_key = key

# Get a list of suppliers from Google based on location
def get_suppliers_from_google(product, location):
    # Base URL for Google Places Text Search API
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    # Parameters for the API request
    params = {
        'query': f"{product} suppliers in {location}",
        'key': api_key,
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        suppliers = data.get('results', [])
        return suppliers
    else:
        print(f"Error: {response.status_code}")
        return []


def get_supplier_details(place_id):
    place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"

    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,geometry,website,rating',
        'key': api_key,
    }

    response = requests.get(place_details_url, params=params)

    if response.status_code == 200:
        details = response.json().get('result', {})
        return details
    else:
        print(f"Error: {response.status_code}")
        return {}


def get_supplier_website(supplier, placeId):
    # Step 1: Get initial suppliers list from Google Places API

    # Step 2: For each supplier, get detailed info using place_id
        place_id = supplier.get('place_id')
        if place_id == placeId:
            return get_supplier_details(place_id).get('website', 'N/A')


def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in values]

def compute_score(distance, rating, weight_distance=0.8, weight_rating=0.2):
    normalized_distance = 1 - distance  # Closer is better
    normalized_rating = rating  # Higher is better
    score = (weight_distance * normalized_distance) + (weight_rating * normalized_rating)
    return score

def calculate_distance(supplier_lat, supplier_lng, org_location):
    supplier_location = (supplier_lat, supplier_lng)
    return geodesic(supplier_location, org_location).km

def filter_and_sort_suppliers(suppliers, org_location):
    distances = []
    ratings = []
    supplier_data = []
    final_supplier_data =[]

    for supplier in suppliers:
        lat = supplier['geometry']['location']['lat']
        lng = supplier['geometry']['location']['lng']
        distance = calculate_distance(lat, lng, org_location)
        rating = supplier.get('rating', 0)  # Default to 0 if no rating

        distances.append(distance)
        ratings.append(rating)

    normalized_distances = normalize(distances)
    normalized_ratings = normalize(ratings)

    for i, supplier in enumerate(suppliers):

        score = compute_score(normalized_distances[i], normalized_ratings[i])


        supplier_data.append({
            'name': supplier.get('name'),
            'address': supplier.get('formatted_address'),
            'lat': supplier['geometry']['location']['lat'],
            'lng': supplier['geometry']['location']['lng'],
            'distance': distances[i],
            'rating': ratings[i],
            'website': get_supplier_website(supplier, supplier.get('place_id')),
            'score': score
        })
    for  supplier in supplier_data:
        if supplier['website'] != 'N/A':
            final_supplier_data.append(supplier)




    sorted_suppliers = sorted(final_supplier_data, key=lambda x: x['score'], reverse=True)

    #Return result in json format
    return json.dumps(sorted_suppliers)


# Code functionality

# Organization's location (example: New Kigali, Rwanda)
organization_location = (-1.94623268784134, 30.067488122865363)

# Example usage
user_prompt = "suppliers iphone 15 pro in nairobi"

# Extract product and location from the prompt
product, city = extract_product_and_city(user_prompt)
print(product, city)

# Get a list of suppliers from google
suppliers = get_suppliers_from_google(product, city)

# Filter and sort suppliers using trained model
sorted_suppliers = filter_and_sort_suppliers(suppliers, organization_location)
print(sorted_suppliers)


#
