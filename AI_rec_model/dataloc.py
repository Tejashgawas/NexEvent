import requests
import csv
import os
import time

# Google Places API details
API_KEY = 'AIzaSyBO3z3IJk55Q5J-Qfye3ENuTFi6neipEjA'  # Replace with your actual API key
TEXT_SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/textsearch/json'
DETAILS_URL = 'https://maps.googleapis.com/maps/api/place/details/json'
PHOTO_URL = 'https://maps.googleapis.com/maps/api/place/photo'

# Types of places you want to search in Goa
queries_to_search = {
    'event venues in Goa': 'venue',
    'caterers in Goa': 'caterer',
    'decorators in Goa': 'decorator'
}

# Folder paths
photos_folder = 'place_photos'  # Folder to save photos
os.makedirs(photos_folder, exist_ok=True)

# Fields to extract for CSV including latitude & longitude
fields = [
    'category', 'displayName.text', 'formattedAddress', 'rating', 'id', 'types',
    'userRatingCount', 'internationalPhoneNumber', 'googleMapsUri', 'websiteUri',
    'photo_url', 'latitude', 'longitude'
]


# Step 1: Fetch data using Text Search API with pagination
def fetch_text_search(query):
    places = []
    params = {
        'query': query,
        'key': API_KEY
    }

    while True:
        response = requests.get(TEXT_SEARCH_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            places.extend(data.get('results', []))

            # Check if there's a next page token
            next_page_token = data.get('next_page_token')
            if next_page_token:
                time.sleep(2)  # Google recommends waiting a couple of seconds
                params['pagetoken'] = next_page_token
            else:
                break  # No more pages
        else:
            print(f"Error fetching data for query '{query}':", response.status_code, response.text)
            break  # Exit if there's an error

    return places


# Step 2: Fetch additional details using Place Details API
def fetch_place_details(place_id):
    params = {
        'place_id': place_id,
        'fields': 'formatted_address,international_phone_number,url,website,rating,photos',
        'key': API_KEY
    }
    response = requests.get(DETAILS_URL, params=params)
    if response.status_code == 200:
        return response.json().get('result', {})
    else:
        print(f"Error fetching details for place_id {place_id}:", response.status_code, response.text)
        return {}


# Step 3: Retrieve photo URL using Place Photo API
def fetch_photo(photo_reference):
    params = {
        'photoreference': photo_reference,
        'maxwidth': 400,  # Specify the desired width of the photo
        'key': API_KEY
    }
    response = requests.get(PHOTO_URL, params=params)
    if response.status_code == 200:
        return response.url
    else:
        print(f"Error fetching photo for reference {photo_reference}: {response.status_code}")
        return None


# Step 4: Categorize places, fetch data, and save to CSV
def extract_and_save_to_csv():
    all_places = []

    for query, category in queries_to_search.items():
        print(f"Fetching data for: {category}...")  # Debugging line
        places = fetch_text_search(query)

        print(f"Found {len(places)} results for {category}.")  # Debugging line
        for place in places:
            place_types = place.get('types', [])

            row = {
                'category': category,  # Set category
                'displayName.text': place.get('name', ''),
                'formattedAddress': place.get('formatted_address', ''),
                'rating': place.get('rating', ''),
                'id': place.get('place_id', ''),
                'types': ', '.join(place_types),
                'userRatingCount': place.get('user_ratings_total', ''),
                'latitude': place.get('geometry', {}).get('location', {}).get('lat', ''),
                'longitude': place.get('geometry', {}).get('location', {}).get('lng', '')
            }

            # Fetch more details from Details API
            if row['id']:
                details = fetch_place_details(row['id'])
                row['internationalPhoneNumber'] = details.get('international_phone_number', '')
                row['googleMapsUri'] = details.get('url', '')
                row['websiteUri'] = details.get('website', '')

                # Get the photo URL if available
                photos = details.get('photos', [])
                if photos:
                    photo_reference = photos[0].get('photo_reference')
                    if photo_reference:
                        row['photo_url'] = fetch_photo(photo_reference)
                    else:
                        row['photo_url'] = ''
                else:
                    row['photo_url'] = ''
            else:
                row['internationalPhoneNumber'] = ''
                row['googleMapsUri'] = ''
                row['websiteUri'] = ''
                row['photo_url'] = ''

            all_places.append(row)

    # Write to CSV
    csv_file_path = 'goa_event_data_combined.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()

        for place in all_places:
            csv_writer.writerow(place)

    print(f"Data successfully saved to {csv_file_path}")


# Run the script
extract_and_save_to_csv()
