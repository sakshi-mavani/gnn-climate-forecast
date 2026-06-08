# Climate Forecasting using Graph Neural Networks (GCN, GAT, GCN-LSTM)

## Overview

This project develops a spatiotemporal climate forecasting system using Graph Neural Networks and Deep Learning models. The objective is to improve forecasting accuracy by modeling both temporal climate patterns and spatial relationships among geographically distributed sensors.

Traditional forecasting methods often ignore interactions between neighboring locations. This project addresses that limitation by representing climate sensors as nodes in a graph and applying Graph Neural Networks to capture spatial dependencies.

The project implements and compares multiple forecasting approaches:

* ARIMA
* LSTM
* Graph Convolutional Network (GCN)
* Graph Attention Network (GAT)
* Hybrid GCN-LSTM

The hybrid GCN-LSTM architecture combines graph-based spatial learning with temporal sequence modeling to achieve the strongest forecasting performance.

---

## Live Deployment

### Production API

**Live API URL**

https://gnn-climate-forecast.onrender.com

### Interactive API Documentation

**Swagger UI**

https://gnn-climate-forecast.onrender.com/docs

**ReDoc Documentation**

https://gnn-climate-forecast.onrender.com/redoc

### Available Endpoints

| Method | Endpoint | Description         |
| ------ | -------- | ------------------- |
| GET    | /        | Project Information |
| GET    | /health  | Health Check        |
| POST   | /predict | Climate Prediction  |

---

## Business and Social Impact

Accurate climate forecasting has significant real-world applications:

* Disaster preparedness and early warning systems
* Agricultural planning and crop management
* Water resource optimization
* Energy demand forecasting
* Climate resilience and sustainability initiatives

This project demonstrates how modern Graph AI techniques can support environmental intelligence systems and improve forecasting reliability.

---

## Research Motivation

Most forecasting models treat climate observations independently. However, climate sensors naturally form a connected network where nearby locations influence each other.

Graph Neural Networks provide a mechanism to model these relationships directly. By incorporating geographic structure into the forecasting process, the model can learn both local and global climate patterns that traditional methods often miss.

---

## Methodology

### Data Processing

The dataset contains climate-related measurements such as:

* Temperature
* Humidity
* Rainfall

Data preprocessing includes:

* Missing value handling
* Feature normalization
* Time-series sequence generation

### Graph Construction

A K-Nearest Neighbor (KNN) graph is constructed where:

* Nodes represent climate sensors
* Edges represent geographic proximity

### Forecasting Models

The following models were implemented and evaluated:

1. ARIMA
2. LSTM
3. GCN
4. GAT
5. GCN-LSTM

### Evaluation Metrics

Performance is evaluated using:

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Prediction vs Actual Analysis

---

## Model Performance

| Model    | RMSE  |
| -------- | ----- |
| ARIMA    | 4.81  |
| LSTM     | 30.28 |
| GCN      | 1.79  |
| GAT      | 0.80  |
| GCN-LSTM | 0.61  |

### Performance Ranking

1. GCN-LSTM
2. GAT
3. GCN
4. ARIMA
5. LSTM

---

## Why GCN-LSTM Performs Better

The GCN-LSTM architecture combines two complementary learning mechanisms.

### Graph Convolutional Network (GCN)

Captures:

* Spatial relationships between sensors
* Geographic dependencies
* Neighbor-to-neighbor climate interactions

### Long Short-Term Memory (LSTM)

Captures:

* Temporal dependencies
* Historical climate trends
* Sequential forecasting patterns

By jointly learning spatial and temporal information, GCN-LSTM significantly improves forecasting accuracy compared to traditional statistical and standalone deep learning models.

---

## Visualizations

The project generates multiple evaluation visualizations.

### Sensor Network Graph

Visual representation of climate sensor connectivity.

### Prediction vs Actual

Comparison between observed and forecasted climate values.

### RMSE Comparison

Performance comparison across all forecasting models.

---

## Deployment Architecture

### FastAPI

REST API endpoints:

* GET /
* GET /health
* POST /predict

### Docker

Containerized deployment for reproducibility and portability.

### Render Cloud Deployment

The forecasting API is deployed on Render and publicly accessible through a production endpoint.

Features:

* Public REST API
* Interactive Swagger Documentation
* Health Monitoring Endpoint
* Containerized Deployment
* Continuous Deployment via GitHub

---

## Example API Request

```json
{
  "temperature": 45,
  "humidity": 79,
  "rainfall": 25
}
```

## Example API Response

```json
{
  "temperature": 45,
  "humidity": 79,
  "rainfall": 25,
  "predicted_value": 51.2
}
```

---

## Project Structure

```text
gnn-climate-forecast/

├── configs/
├── data/
│   └── raw/
├── deployment/
│   ├── app.py
│   └── Dockerfile
├── results/
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Testing

The project includes validation and testing components for:

* Data preprocessing
* Graph construction
* Model execution
* Training pipeline verification
* API endpoint testing

---

## Future Work

Potential future enhancements include:

* Transformer-based climate forecasting
* Dynamic graph learning
* Real-time streaming climate data
* Explainable AI for climate decision support
* Multi-region forecasting systems
* Large-scale environmental intelligence systems

---

## Technical Skills Demonstrated

* Python
* FastAPI
* Graph Neural Networks (GCN, GAT)
* Deep Learning
* LSTM Networks
* PyTorch
* Docker
* REST API Development
* Machine Learning Evaluation
* Data Visualization
* Git & GitHub
* Cloud Deployment (Render)

---

## Resume Highlights

* Developed a hybrid GCN-LSTM model for climate forecasting, outperforming traditional forecasting baselines.
* Implemented Graph Neural Network architectures including GCN and GAT for spatial-temporal climate modeling.
* Built an end-to-end forecasting pipeline covering data preprocessing, graph construction, model training, evaluation, and deployment.
* Designed and deployed a production-ready FastAPI service with interactive API documentation.
* Containerized the application using Docker and deployed it on Render Cloud for public access.

---

## Author

**Sakshi Mavani**

Data Science & Machine Learning Enthusiast

GitHub:
https://github.com/sakshi-mavani

Project Repository:
https://github.com/sakshi-mavani/gnn-climate-forecast
