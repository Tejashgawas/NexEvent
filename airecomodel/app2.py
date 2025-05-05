# app.py (updated loading and scaling)
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from math import radians, cos, sin, asin, sqrt
from difflib import get_close_matches
from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeopyError
from time import sleep
from geopy.distance import geodesic
import googlemaps
GMAPS = googlemaps.Client(key=os.environ["AIzaSyBO3z3IJk55Q5J-Qfye3ENuTFi6neipEjA"])

location_cache = {}


def geocode_with_cache(location):
    if location in location_cache:
        return location_cache[location]

    loc_obj = geolocator.geocode(location, timeout=10)
    if loc_obj:
        coords = (loc_obj.latitude, loc_obj.longitude)
        location_cache[location] = coords
        return coords
    return None


app = Flask(__name__)
# 1) Increase default timeout when you create the geolocator:
geolocator = Nominatim(user_agent="event_recommender", timeout=10)

# Load model & encoders
model = load_model('models/recommender_model.h5')
category_encoder = joblib.load('models/category_encoder.pkl')
types_encoder = joblib.load('models/types_encoder.pkl')

# Load separate scalers
rating_scaler = joblib.load('models/rating_scaler.pkl')
rating_count_scaler = joblib.load('models/rating_count_scaler.pkl')
distance_scaler = joblib.load('models/distance_scaler.pkl')

# Load and preprocess datasets
venues = pd.read_csv('goa_event_venues_updated.csv', encoding='ISO-8859-1')
caterers = pd.read_csv('goa_event_caterers_updated.csv', encoding='ISO-8859-1')
decorators = pd.read_csv('goa_event_decorators_updated.csv', encoding='ISO-8859-1')

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    evt = data.get('event_type', '').strip().lower()
    location = data.get('location', '').strip()

    coords = geocode_with_cache(location)
    if not coords:
        return jsonify({"error": "Location not found"}), 400
    lat, lon = coords

    # ——— Ensure 'types' exists & is clean ———
    venues['types'] = venues.get('types', pd.Series('unknown')).fillna('unknown').astype(str).str.strip().str.lower()
    caterers['types'] = 'unknown'
    decorators['types'] = 'unknown'
    for df_ in (venues, caterers, decorators):
        df_['types'] = df_['types'].replace({
            'marriage': 'wedding',
            'reception': 'wedding',
            'conference': 'corporate'
        })

    # Build list of valid event types (all strings)
    all_types = pd.concat([venues, caterers, decorators])['types'].unique().tolist()

    # Fuzzy match event type
    all_types = pd.concat([venues, caterers, decorators])['types'].unique().tolist()
    match = get_close_matches(evt, all_types, n=1, cutoff=0.6)
    evt_final = match[0] if match else 'unknown'
    evt_idx = types_encoder.transform([evt_final])[0]

    # Prepare combined df & filter venues by type
    df = pd.concat([venues.assign(category='venue'),
                    caterers.assign(category='caterer'),
                    decorators.assign(category='decorator')],
                   ignore_index=True)
    filtered = df[(df['category'] != 'venue') | (df['types'] == evt_final)]

    recs = []
    for _, row in filtered.iterrows():
        cat_idx = category_encoder.transform([row['category']])[0]


        dm = GMAPS.distance_matrix(
            origins=[(lat, lon)],
            destinations=[(row['latitude'], row['longitude'])],
            mode="driving"
        )
        meters = dm['rows'][0]['elements'][0]['distance']['value']
        dist = meters / 1000.0

        dist_df = pd.DataFrame({'distance_km': [dist]})
        count_df = pd.DataFrame({'userRatingCount': [row['userRatingCount']]})

        dist_s = distance_scaler.transform(dist_df)[0][0]
        count_s = rating_count_scaler.transform(count_df)[0][0]
        type_idx = evt_idx if row['category']=='venue' else len(types_encoder.classes_)

        inp = [
            np.array([[cat_idx]]),
            np.array([[type_idx]]),
            np.array([[count_s]]),
            np.array([[dist_s]])
        ]
        score = float(model.predict(inp, verbose=0)[0][0])
        recs.append({
            'name': row['displayName.text'],
            'category': row['category'],
            'rating': row['rating'],
            'distance_km': round(dist,2),
            'score': score,  # ✅ add this
            'normalized_score': None,  # will fill later
            'maps': row.get('googleMapsUri','')
        })

    out = pd.DataFrame(recs)


    # Normalize scores manually
    scores = [r['score'] for r in recs]
    mn, mx = min(scores), max(scores)
    for r in recs:
        r['normalized_score'] = round((r['score']-mn)/(mx-mn),3) if mx>mn else 0.0

    result = {}
    for cat,label in [('venue','🏢 Top Venues'),('caterer','🍽️ Top Caterers'),('decorator','🎨 Top Decorators')]:
        top5 = sorted([r for r in recs if r['category']==cat], key=lambda x: x['normalized_score'], reverse=True)[:5]
        result[label] = top5

    return jsonify(result)

if __name__=='__main__':
    app.run(debug=False)
