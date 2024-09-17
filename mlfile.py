import pandas as pd
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import requests
import spacy
from dotenv import load_dotenv
import os
import json

# Load environment variables from the .env file
load_dotenv()


# Access the API key
key = os.getenv('API_KEY')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Custom function to extract product and location
def extract_product_and_location(prompt):
    doc = nlp(prompt)

    # Rule-based method to extract product by keyword matching
    product = None
    location = None
    product_keywords = ['product', 'supplies', 'supplier', 'item']

    words = prompt.split()  # Split the prompt into words
    for i, word in enumerate(words):
        if word.lower() in product_keywords and i+1 < len(words):
            product = words[i+1]  # Assume the next word is the product

    # Use spaCy for location (GPE = cities, countries, etc.)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE stands for geopolitical entities like cities, countries
            location = ent.text

    # Fallback location detection by looking for capitalized words or common patterns
    if location is None:
        for i, word in enumerate(words):
            if word.lower() == "in" and i+1 < len(words):
                if words[i+1][0].isupper():  # Check if the next word is capitalized (likely a city)
                    location = words[i+1]
                    break

    return product, location

# Fetch real time data using google place API
def get_suppliers_from_google(api_key, product, location):

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        'query': f'{product} suppliers in {location}',
        'key': api_key
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json().get('results', [])
        suppliers = []

        for result in results:
            supplier = {
                'Supplier_Name': result.get('name'),
                'Product_Name': product,
                'Product_Url': result.get('url', 'N/A'),
                'Lat': result['geometry']['location']['lat'],
                'Lng': result['geometry']['location']['lng'],
                'Price': None
            }
            suppliers.append(supplier)

        return suppliers
    else:
        print(f"Error fetching data from Google Places API: {response.status_code}")
        return []

#Calculate Supplier's distance from the org
def calculate_distance(supplier_lat, supplier_lng, org_lat, org_lng):
    """
    Calculate the distance between the supplier and the organization.

    Args:
        supplier_lat (float): Latitude of the supplier.
        supplier_lng (float): Longitude of the supplier.
        org_lat (float): Latitude of the organization.
        org_lng (float): Longitude of the organization.

    Returns:
        float: Distance in kilometers.
    """
    supplier_location = (supplier_lat, supplier_lng)
    org_location = (org_lat, org_lng)
    return haversine(supplier_location, org_location)

# Load the trained model and scaler
with open('supplier_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Organization's location (latitude and longitude)
org_lat = -1.9489776831828343 # Example: Kigali City latitude
org_lng = 30.052381922566386  # Example: Kigali City longitude


#Extract variables from prompts
api_key = key
user_input = "Please a list of suppliers for product laptops in nairobi"
product, location = extract_product_and_location(user_input)


suppliers = get_suppliers_from_google(api_key, product, location)




# Calculate the distance for each supplier
for supplier in suppliers:
    supplier['Distance'] = calculate_distance(supplier['Lat'], supplier['Lng'], org_lat, org_lng)

# Convert the suppliers data into a DataFrame
df_suppliers = pd.DataFrame(suppliers)


df_suppliers['Price'] = df_suppliers['Price']

X_real_time = df_suppliers[['Price', 'Distance']]
X_real_time_scaled = scaler.transform(X_real_time)

# Predict the ranking scores using the trained model
df_suppliers['Score'] = model.predict(X_real_time_scaled)

# Sort the suppliers by the predicted score (lower score = better ranking)
df_suppliers_sorted = df_suppliers.sort_values(by='Score')

# Display the top results
print(df_suppliers_sorted[['Supplier_Name', 'Product_Name', 'Price', 'Distance', 'Score']])


def format_supplier_results(suppliers):
    results = []

    for supplier in suppliers:
        result = {
            "supplier_name": supplier['name'],
            "product_name": supplier['product'],
            "product_link": supplier['link'],
            "price": supplier['price'],
            "location": supplier['location']
        }
        results.append(result)

    # Convert to JSON
    return json.dumps(results, indent=4)

# Example usage to format output for the chatbox
output = format_supplier_results(df_suppliers_sorted)
print(output)  # This would be returned in the chatbox
