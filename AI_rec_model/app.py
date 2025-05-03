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

# Initialize geolocator
geolocator = Nominatim(user_agent="event_recommender")

# 1) Load model & encoders
model = load_model('recommender_model.h5')
category_encoder = joblib.load('category_encoder.pkl')
types_encoder = joblib.load('types_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Determine the index used for “unknown” in training
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

# 3) Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * asin(sqrt(a))

# 4) API route to handle user input and recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
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

    recs = []
    for _, row in df_filtered.iterrows():
        cat = row['category']
        cat_idx = category_encoder.transform([cat])[0]

        dist = haversine(lat, lon, row['latitude'], row['longitude'])
        dist_s = scaler.transform([[dist]])[0][0]
        count_s = scaler.transform([[row['userRatingCount']]])[0][0]

        # Determine type index for venues or set to unknown for others
        type_idx = evt_idx if cat == 'venue' else unknown_type_index

        # Prepare input for model prediction
        inp = [
            np.array([[cat_idx]], dtype='int32'),
            np.array([[type_idx]], dtype='int32'),
            np.array([[count_s]], dtype='float32'),
            np.array([[dist_s]], dtype='float32')
        ]
        score = model.predict(inp, verbose=0)[0][0]

        recs.append({
            'name': row['displayName.text'],
            'category': row['category'],
            'rating': row['rating'],
            'distance_km': round(dist, 2),
            'score': score,
            'maps': row.get('googleMapsUri', '')
        })

    # Rank and normalize scores
    out = pd.DataFrame(recs)
    out['score'] = pd.to_numeric(out['score'], errors='coerce')  # Ensure it's float

    # Normalize the score to a 0–1 range
    out['normalized_score'] = (out['score'] - out['score'].min()) / (out['score'].max() - out['score'].min())
    out['normalized_score'] = out['normalized_score'].round(3)  # Round to 3 decimal places

    # Select and reorder columns for a clean display
    out_display = out[['name', 'category', 'rating', 'distance_km', 'normalized_score', 'maps']]

    # Prepare the result for each category
    result = {}
    for category, label in [('venue', "🏢 Top Venues"),
                            ('caterer', "🍽️ Top Caterers"),
                            ('decorator', "🎨 Top Decorators")]:
        top = out_display[out_display['category'] == category].sort_values(by='normalized_score', ascending=False).head(5)
        result[label] = top[['name', 'rating', 'distance_km', 'normalized_score', 'maps']].to_dict(orient='records')

    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
