# Climate Analysis Model for Canada

## Overview
This repository contains a Climate Analysis Model that predicts daily mean temperatures (`tas`) across Canada. The model leverages machine learning using TensorFlow to analyze both temporal and spatial climate data, providing insights into historical trends and offering future predictions.

## Dataset
- **Source**: Adjusted and Homogenized Canadian Climate Data (AHCCD)  
- **Records**: 158,427 climate records  
- **Features**:  
  - Geographic: Latitude (`lat`), Longitude (`lon`), Elevation (`elev`), Province (`prov`)  
  - Temporal: Year, Day of Year, Date (`time`)  
  - Temperature Metrics: Mean (`tas`), Maximum (`tasmax`), Minimum (`tasmin`)  
- **Data Preprocessing**:  
  - Handled missing values and quality flags  
  - Standardized numeric features (mean=0, std=1)  
  - Extracted temporal patterns such as `day_of_year`

## Model Architecture
- **Type**: Fully Connected Neural Network (FCNN) using TensorFlow and Keras  
- **Layers**:  
  - Input layer: Receives standardized geographic and temporal features  
  - Hidden layers: Dense layers with ReLU activation and dropout for regularization  
  - Output layer: Single neuron with linear activation for regression output

## Model Specifications
- **Optimizer**: Adam (adaptive learning rate optimization)  
- **Loss Function**: Mean Squared Error (MSE)  
- **Evaluation Metric**: Mean Absolute Error (MAE)  
- **Training Configuration**:  
  - 10 epochs  
  - Batch size: 32  
  - 80/20 train-test split  

## Results
- **Training MAE**: 0.26°C  
- **Test MAE**: 2.28°C  

While the training performance is strong, the higher test MAE suggests overfitting could be addressed through further regularization or parameter tuning.

## Features and Tools
- **Python**: Primary programming language  
- **TensorFlow & Keras**: Model building and training  
- **pandas & NumPy**: Data manipulation and preprocessing  
- **scikit-learn**: Scaling, splitting, and evaluation metrics  
- **Matplotlib & Seaborn**: Data visualization  

Feel free to explore the repository and contribute!
