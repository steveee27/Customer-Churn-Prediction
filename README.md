# Customer Churn Prediction

This project predicts customer churn for a bank by analyzing customer data using machine learning techniques. It includes data preprocessing, such as handling missing values and encoding features, and model training with Random Forest and XGBoost to identify patterns associated with churn. The best model is deployed using Streamlit, allowing users to input customer details and receive real-time churn predictions. This solution helps banks proactively address customer retention challenges.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Overview

The **Customer Churn Prediction** project aims to predict whether a customer will churn based on historical data. The project includes:

- Data preprocessing (handling missing values, feature encoding, and scaling).
- Model training with Random Forest and XGBoost classifiers.
- Evaluation of model performance.
- Deployment of the best-performing model using Streamlit.

## Technologies Used

- **Python 3.10**
- **Pandas** for data manipulation.
- **Scikit-learn** for preprocessing and machine learning.
- **XGBoost** for gradient boosting.
- **Streamlit** for creating a web-based UI.
- **Joblib** for model serialization.

## Project Structure

```plaintext
├── data/
│   └── Customer-Dataset.csv      # Dataset
├── src/
│   ├── Training-Model.ipynb      # Notebook for training models
│   ├── Training-Model_OOP.py     # OOP-based training script
├── models/
│   ├── xgb_classifier_model.pkl  # Trained XGBoost model
│   ├── gender_encoder.pkl        # Gender encoder
│   ├── hasCrCard_encoder.pkl     # HasCrCard encoder
│   ├── isActiveMember_encoder.pkl# IsActiveMember encoder
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
└── main.py                       # Streamlit deployment script
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Pipenv or virtualenv for environment management

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/steveee27/Customer-Churn-Prediction.git
    cd Customer-Churn-Prediction
    ```

2. Create a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate    # On Windows: env\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Place the dataset (`Customer-Dataset.csv`) in the `data/` directory.

5. Place the pre-trained models (`.pkl` files) in the `models/` directory.

---

### Usage

#### Model Training

1. Run the OOP-based training script to train the model and save artifacts:
    ```bash
    python src/Training-Model_OOP.py
    ```

2. Ensure that the trained model and encoders are saved in the `models/` directory.

#### Deployment

1. Start the Streamlit app:
    ```bash
    streamlit run main.py
    ```

2. Open the URL provided by Streamlit (e.g., `http://localhost:8501`) in your browser.

#### Prediction

Enter customer data in the web interface to get a prediction on whether the customer will churn.

## Features

- **Data Preprocessing**: Cleans and prepares raw data for modeling, including handling missing values, encoding categorical variables, and scaling numerical features.
- **Model Training**: Implements two machine learning models (Random Forest and XGBoost) with hyperparameter tuning to optimize performance.
- **Evaluation**: Provides detailed classification reports and other metrics to evaluate model performance.
- **Deployment**: Features a user-friendly web application built with Streamlit to predict customer churn interactively.

---

## Contributing

Contributions to the project are welcome! To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Add your message here"
    ```
4. Push your changes to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
5. Create a pull request in the main repository.

For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project as long as proper attribution is given to the original author.
