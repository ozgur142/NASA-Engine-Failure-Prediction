# RUL Prediction using LSTM and Autoencoder

## **Overview**

This project predicts the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA C-MAPSS dataset**. It leverages:

- An **Autoencoder** for feature extraction from sensor data.
- An **LSTM model** for time-series prediction of RUL.

The goal is to estimate how many operational cycles an engine has left before failure, which is critical for predictive maintenance in aviation and other industries.

---

## **Dataset**

The dataset used is from the **CMAPSSData** folder:

- **train_FD001.txt**: Training data containing sensor readings and operational settings.
- **test_FD001.txt**: Test data containing sensor readings and operational settings.
- **RUL_FD001.txt**: Actual remaining useful life for test engines.

---
## Preprocessing

- The dataset is cleaned by removing NaN columns.

- Sensor values are normalized using MinMaxScaler.

- The RUL for training data is computed as:

train_data['RUL'] = train_data.groupby('unit_number')['time_cycles'].transform(max) - train_data['time_cycles']

- The test dataset is aligned with its corresponding RUL values.

---

## **Model Architecture**

### **1. Autoencoder (Feature Extraction)**
- **Input**: 21 sensor features.
- **Encoder**: Reduces the input to an 8-dimensional latent space.
- **Decoder**: Reconstructs the input from the latent space.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.

### **2. LSTM Model (RUL Prediction)**
- **Input**: Encoded sensor features from the Autoencoder.
- **LSTM Layers**: 2 layers with a hidden size of 16.
- **Output**: Predicted RUL.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam.

---

## **Training**

1. **Autoencoder**:
   - Trained for **20 epochs** to learn a compressed representation of the sensor data.

2. **LSTM Model**:
   - Encoded sensor data is used as input.
   - Trained for **80 epochs** with a sequence length of **30**.

---

## **Evaluation**

Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used as evaluation metrics.

MAE = mean_absolute_error(y_true, y_pred)
RMSE = np.sqrt(mean_squared_error(y_true, y_pred))

## **Installation**
Clone this repository and install dependencies:
```bash
pip install -r requirements.txt
```
---
## Future Improvements

Implement attention mechanisms for better LSTM performance.

Fine-tune hyperparameters using Bayesian Optimization.

Explore transformer-based models for RUL prediction.