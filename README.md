# Shipment Sure – Delivery Time Prediction System

**Shipment Sure** is a machine learning-powered web application designed to predict the likelihood of shipment delays. By analyzing over 90 parameters—including supplier reliability, shipping distance, weather conditions, and seasonal factors—it provides real-time actionable insights for supply chain optimization.

## 🚀 Key Features
- **Real-Time Predictions**: Instant classification of orders as "On-Time" or "Delayed".
- **Advanced Telemetry**: Expands user inputs into a 93-feature vector for high-precision model inference.
- **Risk Analysis**: Calculates probability scores and identifies high-risk suppliers or logistics pathways.
- **Interactive UI**: User-friendly dashboard for entering order details and visualizing confidence scores.
- **Robust Backend**: Powered by Flask and a pre-trained Random Forest Classifier.

## 🛠️ Tech Stack
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn (Random Forest), Pandas, NumPy
- **Data Processing**: Feature Engineering, One-Hot Encoding, Heuristic Logistics Modeling

## 🔮 Model Capabilities
The model takes into account:
- Supplier performance history (On-time rates, Lead times)
- Logistics factors (Shipping mode, Carrier, Distance, Region)
- External conditions (Weather, Peak Season, Holidays)
- Computed risk metrics and mathematical transformations for parity with training data.
