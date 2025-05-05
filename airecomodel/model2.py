# model.py (refactored for separate scalers)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import joblib
from math import radians, cos, sin, asin, sqrt

# Load cleaned datasets
venues_df = pd.read_csv('goa_event_venues_updated.csv', encoding='ISO-8859-1')
caterers_df = pd.read_csv('goa_event_caterers_updated.csv', encoding='ISO-8859-1')
decorators_df = pd.read_csv('goa_event_decorators_updated.csv', encoding='ISO-8859-1')

# Add categoryenues_df['category'] = 'venue'
caterers_df['category'] = 'caterer'
decorators_df['category'] = 'decorator'

# Combine
df = pd.concat([venues_df, caterers_df, decorators_df], ignore_index=True)

# Fill and normalize 'types'
df['types'] = df['types'].fillna('unknown').str.strip().str.lower()
df['types'] = df['types'].replace({
    'marriage': 'wedding',
    'reception': 'wedding',
    'conference': 'corporate',
})

# Encode categories and types
category_encoder = LabelEncoder()
df['category_encoded'] = category_encoder.fit_transform(df['category'])

types_encoder = LabelEncoder()
df['types'] = df['types'].astype(str)
df['type_encoded'] = types_encoder.fit_transform(df['types'])

# Initialize separate scalers
rating_scaler = MinMaxScaler()
count_scaler = MinMaxScaler()
distance_scaler = MinMaxScaler()

# Fit each scaler on its column
df['rating_scaled'] = rating_scaler.fit_transform(df[['rating']])
df['user_rating_count_scaled'] = count_scaler.fit_transform(df[['userRatingCount']])

# Haversine function to compute distance

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

# Compute distance from a reference point (e.g., user)
user_lat, user_lon = 15.2993, 74.124

df['distance_km'] = df.apply(
    lambda row: haversine(user_lat, user_lon, row['latitude'], row['longitude']), axis=1
)
df['distance_scaled'] = distance_scaler.fit_transform(df[['distance_km']])

# Handle unknown types for non-venues
unknown_type_index = df['type_encoded'].max() + 1
df['type_encoded'] = df.apply(
    lambda row: row['type_encoded'] if row['category'] == 'venue' else unknown_type_index,
    axis=1
)

# Prepare inputs and target
X = df[[
    'category_encoded',
    'type_encoded',
    'user_rating_count_scaled',
    'distance_scaled'
]].values.astype('float32')
y = df['rating_scaled'].values.astype('float32')

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build DNN with embeddings
input_category = Input(shape=(1,), dtype='int32')
input_type = Input(shape=(1,), dtype='int32')
input_user_rating = Input(shape=(1,), dtype='float32')
input_distance = Input(shape=(1,), dtype='float32')

category_emb = Embedding(input_dim=len(category_encoder.classes_), output_dim=4)(input_category)
type_emb = Embedding(input_dim=unknown_type_index+1, output_dim=4)(input_type)

cat_flat = Flatten()(category_emb)
type_flat = Flatten()(type_emb)

x = Concatenate()([cat_flat, type_flat, input_user_rating, input_distance])
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear')(x)

model = Model(
    inputs=[input_category, input_type, input_user_rating, input_distance],
    outputs=output
)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = model.fit(
    [X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3]],
    y_train,
    validation_data=(
        [X_val[:, 0], X_val[:, 1], X_val[:, 2], X_val[:, 3]],
        y_val
    ),
    epochs=50,
    batch_size=32
)

# Plot and save training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

# Evaluate model
y_pred = model.predict(
    [X_val[:, 0], X_val[:, 1], X_val[:, 2], X_val[:, 3]]
)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\n📊 Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save artifacts
model.save('models/recommender_model.h5')
joblib.dump(category_encoder, 'models/category_encoder.pkl')
joblib.dump(types_encoder, 'models/types_encoder.pkl')
joblib.dump(rating_scaler, 'models/rating_scaler.pkl')
joblib.dump(count_scaler, 'models/rating_count_scaler.pkl')
joblib.dump(distance_scaler, 'models/distance_scaler.pkl')