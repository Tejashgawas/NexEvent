from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
from difflib import get_close_matches
app = Flask(__name__)

# Initialize geolocator
geolocator = Nominatim(user_agent="event_recommender")
# Load assets
import joblib
from tensorflow.keras.models import load_model

# Initialize global variables
model = category_encoder = types_encoder = rating_scaler = user_count_scaler = distance_scaler = None
unknown_type_index = None

def load_resources():
    global model, category_encoder, types_encoder, rating_scaler, user_count_scaler, distance_scaler, unknown_type_index
    if model is None:
        model = load_model('recommender_model.h5')
        category_encoder = joblib.load('category_encoder.pkl')
        types_encoder = joblib.load('types_encoder.pkl')
        rating_scaler = joblib.load('rating_scaler.pkl')
        user_count_scaler = joblib.load('user_count_scaler.pkl')
        distance_scaler = joblib.load('distance_scaler.pkl')
        unknown_type_index = len(types_encoder.classes_)



# 2) Load & preprocess each dataset
venues = pd.read_csv('goa_event_venues_updated.csv', encoding='ISO-8859-1')
caterers = pd.read_csv('goa_event_caterers_updated.csv', encoding='ISO-8859-1')
decorators = pd.read_csv('goa_event_decorators_updated.csv', encoding='ISO-8859-1')

# Fill missing 'types' for venues and set default for caterers and decorators
venues['types'] = venues['types'].fillna('unknown').str.strip().str.lower()
caterers['types'] = 'unknown'
decorators['types'] = 'unknown'

# Replace synonyms for event types (e.g., 'marriage' to 'wedding')
for df in (venues, caterers, decorators):
    df['types'] = df['types'].replace({
        'marriage': 'wedding',
        'reception': 'wedding',
        'conference': 'birthday'
    })
    # Assign 'category' based on dataframe
    df['category'] = ('venue' if df is venues
                      else 'caterer' if df is caterers
                      else 'decorator')

# Concatenate all dataframes (venues, caterers, decorators)
df = pd.concat([venues, caterers, decorators], ignore_index=True)


# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r



@app.route('/recommend', methods=['POST'])
def recommend():
    load_resources()
    data = request.get_json()

    # Get user input for event type and location
    evt = data.get('event_type', '').strip().lower()
    location = data.get('location', '').strip()

    # Geocode location to get latitude and longitude
    try:
        location_obj = geolocator.geocode(location)
        if location_obj:
            lat = location_obj.latitude
            lon = location_obj.longitude
        else:
            return jsonify({"error": "Location not found"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

        # Fuzzy-match to find the closest event type
    all_types = df['types'].unique().tolist()
    match = get_close_matches(evt, all_types, n=1, cutoff=0.6)
    evt_final = match[0] if match else 'unknown'
    evt_idx = types_encoder.transform([evt_final])[0]

    # Filter venues based on matching event type
    df_filtered = []
    for _, row in df.iterrows():
        if row['category'] == 'venue' and row['types'] != evt_final:
            continue  # Skip venues that don't match the event type
        df_filtered.append(row)

    df_filtered = pd.DataFrame(df_filtered)

    recommendations = []
    for _, row in df_filtered.iterrows():
        cat = row['category']
        cat_idx = category_encoder.transform([cat])[0]

        dist = haversine(lat, lon, row['latitude'], row['longitude'])
        dist_s = distance_scaler.transform(pd.DataFrame({'distance_km': [dist]}))[0][0]
        count_s = user_count_scaler.transform(pd.DataFrame({'userRatingCount': [row['userRatingCount']]}))[0][0]
        # Determine type index for venues or set to unknown for others
        type_idx = evt_idx if cat == 'venue' else unknown_type_index



        pred_input = [
            np.array([cat_idx]),
            np.array([type_idx]),
            np.array([count_s]),
            np.array([dist_s])
        ]

        predicted_rating_scaled = model.predict(pred_input)[0][0]
        predicted_rating = rating_scaler.inverse_transform([[predicted_rating_scaled]])[0][0]

        recommendations.append({
            'name': row['displayName.text'],
            'address': row['formattedAddress'],
            'rating': row['rating'],
            'predicted_rating': round(predicted_rating, 2),
            'distance_km': round(dist, 2),
            'map_link': row['googleMapsUri'],
            'phone': row['internationalPhoneNumber'],
            'photo_url': row['photo_url']
        })

    top_recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)[:5]
    return jsonify(top_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
