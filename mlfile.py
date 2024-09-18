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
    # Regex pattern to capture product and city
    product_pattern = re.search(r'for (.*?) in', prompt)
    city_pattern = re.search(r'in (.*)', prompt)

    product = product_pattern.group(1) if product_pattern else None
    city = city_pattern.group(1) if city_pattern else None

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

    sorted_suppliers = sorted(supplier_data, key=lambda x: x['score'], reverse=True)

    #Return result in json format
    return json.dumps(sorted_suppliers)


# Code functionality

# Organization's location (example: New Kigali, Rwanda)
organization_location = (-1.94623268784134, 30.067488122865363)

# Example usage
user_prompt = "I want a list of suppliers for stationery in Rwanda"

# Extract product and location from the prompt
product, city = extract_product_and_city(user_prompt)

# Get a list of suppliers from google
suppliers = get_suppliers_from_google(product, city)

# Filter and sort suppliers using trained model
sorted_suppliers = filter_and_sort_suppliers(suppliers, organization_location)


print(sorted_suppliers)

#































# # Example usage
# suppliers = get_suppliers_from_google(product, city)
# print(suppliers)

# # Display the returned suppliers with all information
# # for supplier in suppliers:
# #     print(f"Name: {supplier.get('name')}")
# #     print(f"Address: {supplier.get('formatted_address')}")
# #     print(f"Latitude: {supplier['geometry']['location']['lat']}")
# #     print(f"Longitude: {supplier['geometry']['location']['lng']}")
# #     print(f"Website: {supplier.get('website', 'N/A')}")
# #     print(f"Price Level: {supplier.get('price_level', 'N/A')}")
# #     print(f"Rating: {supplier.get('rating', 'N/A')}")
# #     print(f"Business Status: {supplier.get('business_status', 'N/A')}")
# #     print("----------")

# # Assuming the organization's coordinates are known
# organization_location = (-1.9434876880309504, 30.05787508631147)  # Kigali Rwanda



# def compute_score(distance, price_level, weight_distance=0.8, weight_price=0.9):
#     # Normalize and calculate a weighted score (lower score is better)
#     return weight_distance * distance + weight_price * price_level

# def filter_and_sort_suppliers(suppliers, org_location):
#     supplier_data = []

#     for supplier in suppliers:
#         lat = supplier['geometry']['location']['lat']
#         lng = supplier['geometry']['location']['lng']
#         distance = calculate_distance(lat, lng, org_location)
#         price_level = supplier.get('price_level', 3)  # Default to mid-price if not provided

#         score = compute_score(distance, price_level)

#         supplier_data.append({
#             'name': supplier.get('name'),
#             'address': supplier.get('formatted_address'),
#             'lat': lat,
#             'lng': lng,
#             'distance': distance,
#             'price_level': price_level,
#             'rating': supplier.get('rating', 'N/A'),
#             'website': supplier.get('website', 'N/A'),
#             'score': score
#         })

#     # Sort suppliers by score (ascending)
#     sorted_suppliers = sorted(supplier_data, key=lambda x: x['score'])
#     return sorted_suppliers

# # Example usage
# sorted_suppliers = filter_and_sort_suppliers(suppliers, organization_location)

# # Display sorted suppliers
# for supplier in sorted_suppliers:
#     print(f"Supplier Name: {supplier['name']}")
#     print(f"Distance: {supplier['distance']:.2f} km")
#     print(f"Price Level: {supplier['price_level']}")
#     print(f"Score: {supplier['score']:.2f}")
#     print(f"Website: {supplier['website']}")
#     print(f"Rating: {supplier['rating']}")

#     print("------------")






# # Fetch real time data using google place API
# # def get_suppliers_from_google(api_key, product, location):

# #     url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
# #     params = {
# #         'query': f'{product} suppliers in {location}',
# #         'key': api_key
# #     }

# #     response = requests.get(url, params=params)

# #     if response.status_code == 200:
# #         results = response.json().get('results', [])
# #         suppliers = []

#         for result in results:
#             supplier = {
#                 'Supplier_Name': result.get('name'),
#                 'Product_Name': product,
#                 # 'Product_Url': result.get('url', 'N/A'),
#                 'Lat': result['geometry']['location']['lat'],
#                 'Lng': result['geometry']['location']['lng'],
#                 # 'Price': 100000
#             }
#             suppliers.append(supplier)

#         return suppliers
#     else:
#         print(f"Error fetching data from Google Places API: {response.status_code}")
#         return []

# #Calculate Supplier's distance from the org
# def calculate_distance(supplier_lat, supplier_lng, org_lat, org_lng):
#     """
#     Calculate the distance between the supplier and the organization.

#     Args:
#         supplier_lat (float): Latitude of the supplier.
#         supplier_lng (float): Longitude of the supplier.
#         org_lat (float): Latitude of the organization.
#         org_lng (float): Longitude of the organization.

#     Returns:
#         float: Distance in kilometers.
#     """
#     supplier_location = (supplier_lat, supplier_lng)
#     org_location = (org_lat, org_lng)
#     return haversine(supplier_location, org_location)

# # Load the trained model and scaler
# with open('supplier_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Organization's location (latitude and longitude)
# org_lat = -1.9489776831828343 # Example: Kigali City latitude
# org_lng = 30.052381922566386  # Example: Kigali City longitude


# #Extract variables from prompts
# api_key = key
# user_input = "Please a list of suppliers for product laptops in nairobi"
# product, location = extract_product_and_location(user_input)


# suppliers = get_suppliers_from_google(api_key, product, location)

# print(suppliers)




# # Calculate the distance for each supplier
# for supplier in suppliers:
#     supplier['Distance'] = calculate_distance(supplier['Lat'], supplier['Lng'], org_lat, org_lng)

# # Convert the suppliers data into a DataFrame
# df_suppliers = pd.DataFrame(suppliers)



# X_real_time = df_suppliers[['Distance']]
# X_real_time_scaled = scaler.transform(X_real_time)

# # Predict the ranking scores using the trained model
# df_suppliers['Score'] = model.predict(X_real_time_scaled)

# # Sort the suppliers by the predicted score (lower score = better ranking)
# df_suppliers_sorted = df_suppliers.sort_values(by='Score')

# # Display the top results
# print(df_suppliers_sorted[['Supplier_Name', 'Product_Name', 'Distance', 'Score']])


# def format_supplier_results(suppliers):
#     results = []

#     for supplier in suppliers:
#         result = {
#             "supplier_name": supplier['name'],
#             "product_name": supplier['product'],
#             # "product_link": supplier['link'],
#             # "price": supplier['price'],
#             "location": supplier['location']
#         }
#         results.append(result)

#     # Convert to JSON
#     return json.dumps(results, indent=4)

# # Example usage to format output for the chatbox
# output = format_supplier_results(df_suppliers_sorted)
# print(output)  # This would be returned in the chatbox
