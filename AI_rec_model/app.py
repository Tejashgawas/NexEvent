import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from math import radians, cos, sin, asin, sqrt
from difflib import get_close_matches
from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim

# Initialize Flask app
app = Flask(__name__)
geolocator = Nominatim(user_agent="event_recommender")

# Load trained model and preprocessing tools
model = load_model('recommender_model.h5')
category_encoder = joblib.load('category_encoder.pkl')
types_encoder = joblib.load('types_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Index for unknown type (used for caterers/decorators)
unknown_type_index = len(types_encoder.classes_)

# Load datasets
venues = pd.read_csv('goa_event_venues_updated.csv', encoding='ISO-8859-1')
caterers = pd.read_csv('goa_event_caterers_updated.csv', encoding='ISO-8859-1')
decorators = pd.read_csv('goa_event_decorators_updated.csv', encoding='ISO-8859-1')

# Preprocess types
venues['types'] = venues['types'].fillna('unknown').str.strip().str.lower()
caterers['types'] = 'unknown'
decorators['types'] = 'unknown'

# Replace known synonyms
for df in [venues, caterers, decorators]:
    df['types'] = df['types'].replace({
        'marriage': 'wedding',
        'reception': 'wedding',
        'conference': 'corporate'
    })

# Add category column
venues['category'] = 'venue'
caterers['category'] = 'caterer'
decorators['category'] = 'decorator'

# Combine all data
df = pd.concat([venues, caterers, decorators], ignore_index=True)

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    try:
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 6371 * 2 * asin(sqrt(a))
    except Exception:
        return np.nan

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    event_type = data.get('event_type', '').strip().lower()
    location = data.get('location', '').strip()

    # Validate input
    if not location:
        return jsonify({"error": "Location is required"}), 400

    try:
        location_obj = geolocator.geocode(location)
        if not location_obj:
            return jsonify({"error": "Location not found"}), 400
        user_lat = location_obj.latitude
        user_lon = location_obj.longitude
    except Exception as e:
        return jsonify({"error": f"Geolocation error: {str(e)}"}), 500

    # Match closest event type
    all_types = df['types'].unique().tolist()
    matched = get_close_matches(event_type, all_types, n=1, cutoff=0.6)
    final_event_type = matched[0] if matched else 'unknown'
    final_event_type_encoded = types_encoder.transform([final_event_type])[0]

    records = []
    for _, row in df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue

        if row['category'] == 'venue' and row['types'] != final_event_type:
            continue  # Filter out unmatched venues

        try:
            cat_idx = category_encoder.transform([row['category']])[0]
        except:
            continue

        # Distance
        dist = haversine(user_lat, user_lon, row['latitude'], row['longitude'])
        if np.isnan(dist):
            continue

        # Normalize distance and user rating count
        try:
            dist_scaled, count_scaled = scaler.transform([[row['rating'], row['userRatingCount'], dist]])[0][2], scaler.transform([[row['rating'], row['userRatingCount'], dist]])[0][1]
        except:
            dist_scaled, count_scaled = 0, 0

        # Use correct type encoding
        type_idx = final_event_type_encoded if row['category'] == 'venue' else unknown_type_index

        records.append({
            'name': row.get('displayName.text', ''),
            'category': row['category'],
            'rating': row['rating'],
            'distance_km': round(dist, 2),
            'maps': row.get('googleMapsUri', ''),
            'input': [
                np.array([cat_idx], dtype='int32'),
                np.array([type_idx], dtype='int32'),
                np.array([count_scaled], dtype='float32'),
                np.array([dist_scaled], dtype='float32')
            ]
        })

    # Batch prediction
    if not records:
        return jsonify({"error": "No suitable services found"}), 404

    inputs = [
        np.array([r['input'][0] for r in records]),
        np.array([r['input'][1] for r in records]),
        np.array([r['input'][2] for r in records]),
        np.array([r['input'][3] for r in records]),
    ]
    preds = model.predict(inputs, verbose=0).flatten()

    # Append predictions
    for i, r in enumerate(records):
        r['score'] = preds[i]

    df_result = pd.DataFrame(records)
    df_result['normalized_score'] = (
        (df_result['score'] - df_result['score'].min()) / 
        (df_result['score'].max() - df_result['score'].min())
    ).round(3)

    df_result.sort_values(by='normalized_score', ascending=False, inplace=True)

    # Organize by category
    result = {}
    for cat, label in [('venue', '🏢 Top Venues'),
                       ('caterer', '🍽️ Top Caterers'),
                       ('decorator', '🎨 Top Decorators')]:
        top_cat = df_result[df_result['category'] == cat].head(5)
        result[label] = top_cat[['name', 'rating', 'distance_km', 'normalized_score', 'maps']].to_dict(orient='records')

    return jsonify(result)

# Run Flask server
if __name__ == '__main__':
    app.run(debug=False)
