import pandas as pd
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import requests
import spacy

key = 'AIzaSyCsrPonxZ-MLhBcEpGDG6ol6JHZi1szbZM'

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


def get_suppliers_from_google(api_key, product, location):
    """
    Fetch suppliers from Google Places API based on the product and location.

    Args:
        api_key (str): Your Google Places API key.
        product (str): The product to search for.
        location (str): The location to search within.

    Returns:
        List[dict]: A list of suppliers with relevant information.
    """
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
                'Price': None  # You can manually input or scrape prices
            }
            suppliers.append(supplier)

        return suppliers
    else:
        print(f"Error fetching data from Google Places API: {response.status_code}")
        return []


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



# Get real-time suppliers from Google Places API using the extracted variables
api_key = key  # Replace with your Google API key
user_input = "Please a list of suppliers for product laptops in nairobi"
product, location = extract_product_and_location(user_input)


suppliers = get_suppliers_from_google(api_key, product, location)




# Calculate the distance for each supplier
for supplier in suppliers:
    supplier['Distance'] = calculate_distance(supplier['Lat'], supplier['Lng'], org_lat, org_lng)

# Convert the suppliers data into a DataFrame
df_suppliers = pd.DataFrame(suppliers)

# Add a dummy price if the price data is not available (you can also fetch this from another source)
df_suppliers['Price'] = df_suppliers['Price']  # Replace with real prices if available

# Prepare the features (Price and Distance) for scaling
X_real_time = df_suppliers[['Price', 'Distance']]
X_real_time_scaled = scaler.transform(X_real_time)

# Predict the ranking scores using the trained model
df_suppliers['Score'] = model.predict(X_real_time_scaled)

# Sort the suppliers by the predicted score (lower score = better ranking)
df_suppliers_sorted = df_suppliers.sort_values(by='Score')

# Display the top results
print(df_suppliers_sorted[['Supplier_Name', 'Product_Name', 'Price', 'Distance', 'Score']])

def format_supplier_results(suppliers):
    """
    Format the sorted suppliers for display in the chatbox.

    Args:
        suppliers (pd.DataFrame): DataFrame containing sorted suppliers.

    Returns:
        str: Formatted string to display in the chatbox.
    """
    results = ""
    for idx, row in suppliers.iterrows():
        results += (f"Supplier: {row['Supplier_Name']}\n"
                    f"Product: {row['Product_Name']}\n"
                    f"Price: {row['Price']}\n"
                    f"Distance: {round(row['Distance'], 2)} km\n"
                    f"Link: {row['Product_Url']}\n\n")

    return results

# Example usage to format output for the chatbox
output = format_supplier_results(df_suppliers_sorted)
print(output)  # This would be returned in the chatbox
