# 🧠 AI-Powered Event Service Recommendation Model

A machine learning-based recommendation engine for **event organizers** to get the best **venue**, **caterer**, and **decorator** suggestions based on user reviews, ratings, and location.

Built as a core module of the **NexEvent** project.

---

## 🎯 Objective

Help users efficiently discover and choose the top-rated event service providers in their locality using AI.  
Key goals:
- Rank services (Venue, Caterers, Decorators) based on rating and popularity
- Incorporate user preferences and location filtering
- Deliver top 3–5 smart recommendations using ML

---

## ⚙️ Features

✅ Predicts top-rated services based on:
- `Average Rating`
- `Number of Reviews`
- `Service Type` (venue, caterer, etc.)
- `Location` (city-based filter)

✅ Real-time recommendation API  
✅ Integrated with NexEvent frontend  
✅ Clean, preprocessed datasets for training  

---
### 🧪 Tech Stack
| Component        | Tech Used                |
| ---------------- | ------------------------ |
| Programming Lang | Python                   |
| ML Framework     | TensorFlow, scikit-learn |
| Backend API      | Flask                    |
| Data Handling    | Pandas, NumPy            |
| Deployment       | flask server             |
| Frontend         | Reactjs,Typescript,Vue   |

---

### 🧪 Running Locally
# 1. Clone the repo
git clone https://github.com/tejashgawas/event-recommendation-model.git
cd event-recommendation-model

# 2. Install requirements
pip install -r requirements.txt

# 3. Run the Flask server
python app.py

# 4. Hit API
http://localhost:5000/recommend?type=venue&city=Goa


## 🏗 Architecture Overview

```text
📦 Dataset (CSV)
   └── service_name, rating, user_rating_count, service_type, city

⬇️

🔄 Preprocessing:
   - Normalize ratings
   - Encode service types
   - One-hot city encoding

⬇️

🧠 TensorFlow Deep Neural Network Model combining popularity and geolocation based filtering
   - Input: [rating, review_count, city, type]
   - Hidden layers: Dense (ReLU)
   - Output: Predicted score for the recommendation

⬇️

📡 Flask API Endpoint:
   /recommend?event_type=Wedding&city=Goa

⬇️

📱 NexEvent Frontend UI:
   - Fetches recommendations
   - Displays top results
---




