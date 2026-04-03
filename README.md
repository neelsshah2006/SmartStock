# SmartStock 📈

ML-powered retail demand forecasting and automated ordering halt decisioning system.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)](https://xgboost.readthedocs.io/)

## What the Project Does

SmartStock is an intelligent retail analytics platform that combines machine learning with real-time data processing to help retailers optimize their inventory management. The system provides:

- **Sales Forecasting**: Predict future daily sales for individual stores using XGBoost regression models
- **Halt Decisioning**: Automatically determine when to halt reordering based on predicted sales and historical patterns
- **Data Management**: Store and manage historical sales data with comprehensive validation
- **Interactive Dashboard**: User-friendly web interface for data entry and predictions

## Why the Project is Useful

Retailers face significant challenges in balancing inventory costs with stockouts. SmartStock addresses these pain points by:

- **Reducing Overstock**: Prevent excess inventory through accurate demand forecasting
- **Minimizing Stockouts**: Ensure product availability with intelligent halt recommendations
- **Data-Driven Decisions**: Replace manual guesswork with ML-powered insights
- **Scalable Architecture**: Handle multiple stores with consistent, automated decision-making
- **Real-Time Processing**: Make predictions on-demand with historical context

## How Users Can Get Started

### Prerequisites

- Python 3.12 or higher
- MongoDB instance (local or cloud)
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/neelsshah2006/SmartStock.git
   cd SmartStock
   ```

2. **Set up the backend**

   ```bash
   cd backend
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Set up the frontend**

   ```bash
   cd ../frontend
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

4. **Configure MongoDB**

   Create a `.env` file in the `backend` directory:

   ```env
   MONGODB_URL=mongodb://localhost:27017/smartstock
   # Or for MongoDB Atlas:
   # MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/smartstock
   ```

5. **Train the models** (optional - pre-trained models included)
   ```bash
   cd backend
   python train_models.py
   ```

### Running the Application

1. **Start the backend API**

   ```bash
   cd backend
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

2. **Start the frontend**

   ```bash
   cd frontend
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   streamlit run app.py
   ```

   The web app will be available at `http://localhost:8501`

### Usage Examples

#### Adding Historical Data

Use the "Add New Data" page in the Streamlit app to input historical sales data:

```python
# Example data record
{
    "store": 1,
    "sale_date": "2023-07-15",
    "day_of_week": 5,
    "sales": 4500.50,
    "promo": 1,
    "school_holiday": 0,
    "state_holiday": "0",
    "competition_distance": 1270.0,
    "store_type": "a",
    "assortment": "a"
}
```

#### Making Predictions

Use the "Predict Sales" page to forecast future sales and get halt recommendations:

```python
# API request example
POST http://localhost:8000/predict
{
    "store": 1,
    "target_date": "2023-07-20",
    "promo": 0,
    "school_holiday": 0,
    "state_holiday": "0"
}
```

## Project Structure

```
SmartStock/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # API entry point
│   │   ├── database/       # MongoDB connection
│   │   ├── models/         # Pre-trained ML models
│   │   ├── routes/         # API endpoints
│   │   ├── schemas/        # Pydantic models
│   │   ├── services/       # Business logic
│   │   └── utils/          # Helper functions
│   ├── data/               # Training data
│   └── requirements.txt
├── frontend/                # Streamlit web app
│   ├── components/          # UI components
│   ├── pages/              # App pages
│   └── requirements.txt
├── data/                   # Sample datasets
├── SmartStock.ipynb        # Model development notebook
└── README.md
```

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.
