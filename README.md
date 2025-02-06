# Resume Categorization

## Overview

This project addresses the challenge of automating resume categorization using machine learning and deep learning techniques.  The goal is to streamline the recruitment process by predicting the most appropriate job category for a given resume, saving recruiters valuable time and resources.

## Key Features

- **Automated Resume Categorization:** Utilizes trained machine learning and deep learning models to predict job categories from resume text.
- **Multiple Model Predictions:** Provides predictions from K-Nearest Neighbors (KNN), Support Vector Machine (SVC), Random Forest, Multilayer Perceptron (MLP), and an ensemble Voting Classifier.
- **Streamlit Application:** Deployed as an interactive and shareable web application using Streamlit.
- **Text Preprocessing:** Implements robust text preprocessing techniques to clean and prepare resume data for model training.
- **Easy Deployment:** Trained models and related artifacts are serialized (pickled) for seamless integration into the Streamlit application.

## Dataset

The project leverages the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) from Kaggle. This dataset contains resumes categorized into 25 distinct job fields, providing a diverse training set for the models.

## Technologies Used

- **Python:** Primary programming language.
- **Pandas:** Data manipulation and analysis.
- **NumPy:** Numerical computing.
- **Scikit-learn:** Machine learning algorithms and tools (TF-IDF, KNN, SVC, Random Forest, LabelEncoder, Train/Test Split, etc.).
- **TensorFlow/Keras:** Deep learning models (MLP, RNN, LSTM, BI-LSTM).
- **Streamlit:** Web application framework for deployment.
- **Pickle:** Serialization of Python objects.
- **Matplotlib:** Visualization.

## Model Development

The following models were developed and trained:

### Machine Learning Models:

- **K-Nearest Neighbors (KNN):** A simple classification algorithm using distance-based classification.
- **Support Vector Machine (SVC):**  Finds an optimal hyperplane to separate classes.
- **Random Forest:** An ensemble method using multiple decision trees.

### Deep Learning Models:

- **Multilayer Perceptron (MLP):** A neural network with hidden layers for complex pattern learning.
- **Recurrent Neural Network (RNN):** Deep learning model using gradient descent or adam optimization.
- **Long Short-Term Memory (LSTM):** Deep learning model.
- **Bidirectional LSTM (BI-LSTM):** Deep learning model using gradient descent.

### Ensemble Model

- **Voting Classifier:** Combines the predictions of KNN, SVC, and Random Forest for improved accuracy.

## Data Preprocessing Steps

1. **Text Cleaning:**
   - Removed URLs, RTs, hashtags, and mentions.
   - Eliminated special characters and non-ASCII characters.
   - Collapsed extra whitespace.

2. **Category Encoding:**
   - Transformed textual job categories into numerical labels using Label Encoding.

3. **TF-IDF Vectorization:**
   - Converted cleaned text data into numerical vectors using TF-IDF.

4. **Data Splitting:**
   - Divided the dataset into training and testing sets.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone [repository_url]
   cd [repository_directory]
