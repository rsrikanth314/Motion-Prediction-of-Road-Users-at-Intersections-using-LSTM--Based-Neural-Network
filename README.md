# Motion Prediction of Road Users at Intersections
## Using LSTM-Based Neural Network

This repository contains the code and related materials for the project on motion prediction of road users at intersections using a Long Short-Term Memory (LSTM)-based neural network.

## Overview

With the increasing demand for intelligent autonomous systems on roads and in human environments, the capacity of such systems to sense, analyze, and predict behavior is highly important. This project focuses on the motion prediction of road users, particularly at intersections, using a neural network-based approach, specifically LSTM. The project provides a multivariate, multi-output, multi-step LSTM model to forecast the future positions of dynamic agents.

## Directory Structure

The project directory has the following structure:

- `data_processing`: Contains scripts or notebooks for data preprocessing.
- `evaluation`: Includes scripts or notebooks for evaluating the performance of the prediction models.
- `pictures_main`: Contains any visual materials or images related to the main project.
- `prediction_models`: Consists of the main scripts or notebooks for training and testing the LSTM prediction models.
- `visualization`: Includes scripts or notebooks for visualizing the results.
- `.DS_Store`: macOS system file storing custom attributes of a folder.
- `main.ipynb`: The main Jupyter notebook containing the overall project implementation.
- `NN_prediction_result_Model_1_TestingID_28.xlsx`: Excel file containing prediction results for Model 1 on testing data.
- `NN_prediction_result_Model_2_TestingID_28.xlsx`: Excel file containing prediction results for Model 2 on testing data.
- `Previously_trained_Model_1.h5`: Previously trained weights for Model 1.
- `Previously_trained_Model_2.h5`: Previously trained weights for Model 2.

## Getting Started

To replicate or extend the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/[your-username]/motion-prediction.git



Usage
The project involves several components:

Data Processing: Check the data_processing directory for scripts or notebooks related to data preprocessing.

Prediction Models: The core of the project is in the prediction_models directory, where LSTM models are trained and tested for motion prediction.

Evaluation: Evaluate the performance of the models using scripts or notebooks in the evaluation directory.

Visualization: Visualize the results with scripts or notebooks found in the visualization directory.

Results: Review the prediction results in the Excel files (NN_prediction_result_Model_1_TestingID_28.xlsx and NN_prediction_result_Model_2_TestingID_28.xlsx).

Results
The project has demonstrated a slight improvement in forecasting the behavior of traffic participants using the LSTM model. Detailed results and analysis can be found in the NN_prediction_result_Model_1_TestingID_28.xlsx and NN_prediction_result_Model_2_TestingID_28.xlsx files.

Contributing
Feel free to contribute to the project by following these steps:

Fork the repository.

Create a new branch:

bash
Copy code
git checkout -b feature-name
Commit your changes:

bash
Copy code
git commit -m 'Add new feature'
Push to the branch:

bash
Copy code
git push origin feature-name
Submit a pull request.

License
This project is licensed under the MIT License.

vbnet
Copy code

Please replace `[your-username]` with your actual GitHub username. This README provides detailed information about the project structure, components, and how to get started with the code. Adjust any details as needed for your specific project.





