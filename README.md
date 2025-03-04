# Spotify Music Genre Classification & Popularity Prediction Using Machine Learning
## Introduction
Music streaming platforms like Spotify rely on data-driven insights to recommend songs, categorize genres, and predict song popularity. With millions of tracks available, automating genre classification and forecasting popularity scores can significantly improve recommendation algorithms and user experience.

This project explores how Machine Learning (ML) models can be used to classify songs into their respective genres and predict their popularity scores based on key audio features like tempo, loudness, and danceability. By comparing different classification and regression models, we aim to find the most accurate and efficient approach for analyzing music trends.

## Project Objectives
🔹 Classify songs into genres using machine learning classification models.
🔹 Predict song popularity scores using regression techniques.
🔹 Enhance recommendation systems by improving genre detection accuracy.
🔹 Compare multiple ML models and optimize their performance.

## Dataset Overview
The dataset used in this project consists of Spotify music tracks with various attributes that describe their audio characteristics. The primary features include:

🎼 Tempo (bpm) – Measures the speed of the song.
🎶 Loudness (dB) – Indicates the volume and intensity of the track.
🕺 Danceability – Represents how suitable a song is for dancing.
⚡ Energy – Captures the overall intensity of the track.
🎤 Speechiness – Determines the presence of spoken words.
📊 Popularity Score – A numeric rating that reflects the song’s popularity.
🎵 Genre – The category of music, such as pop, rock, or jazz.

To ensure high-quality predictions, the dataset underwent cleaning and preprocessing, including removal of irrelevant columns, handling missing values, and feature encoding for categorical attributes. Standardization techniques were applied to numerical features to maintain consistency across different scales.

## Approach & Methodology
This project is divided into two key tasks:

1️⃣ Music Genre Classification
The goal of this task is to automate the categorization of songs into their respective genres. Since genres are categorical labels, this problem falls under classification models in machine learning.

Multiple classification algorithms were tested to determine which one could best identify a song’s genre based on its audio attributes. The models compared include:

✔ Random Forest Classifier – An ensemble learning method that constructs multiple decision trees to improve accuracy and reduce overfitting.
✔ Support Vector Machine (SVM) – A powerful classifier that separates data points using hyperplanes.
✔ K-Nearest Neighbors (KNN) – A distance-based model that classifies songs based on their similarity to nearby tracks.

The performance of these models was evaluated using accuracy metrics and confusion matrices to assess how well they classified songs into genres. Feature selection and hyperparameter tuning were conducted to improve the accuracy and efficiency of the models.

2️⃣ Predicting Song Popularity
The second part of the project focused on predicting a song’s popularity score, which is a continuous numerical value. This task is best suited for regression models, as they estimate numerical outputs based on input features.

Several regression techniques were tested to predict how popular a song will be based on its audio properties. The models evaluated include:

✔ Random Forest Regressor – A robust model that averages multiple decision trees to improve predictions.
✔ Support Vector Regressor (SVR) – A model that uses hyperplanes to make continuous predictions.
✔ Linear Regression – A simple and interpretable model that identifies relationships between features and the popularity score.

The models were compared using Root Mean Squared Error (RMSE), a standard metric that measures the difference between predicted and actual popularity scores. Hyperparameter tuning and cross-validation were applied to optimize the regression models for better accuracy.

## Results & Key Findings
📌 Best Classification Model: Random Forest Classifier achieved the highest accuracy in classifying genres, making it the most effective model for this task.
📌 Best Regression Model: Random Forest Regressor provided the most accurate predictions for song popularity scores, outperforming other regression techniques.
📌 Feature Importance Analysis: Attributes like energy, loudness, and danceability were found to have the strongest impact on both genre classification and popularity prediction.
📌 Hyperparameter Tuning: Fine-tuning parameters significantly improved the performance of all models, proving that optimizing ML models is crucial for real-world applications.

## Challenges & Limitations
While the results were promising, several challenges were encountered:

❌ Imbalanced Dataset: Some genres were underrepresented, which affected classification accuracy.
❌ Subjectivity of Popularity Scores: Popularity is influenced by external factors like marketing and social trends, which are not included in the dataset.
❌ Computational Complexity: Advanced models like SVM and Random Forest require significant computational resources for training and tuning.

Future improvements could involve data augmentation techniques to balance the dataset and deep learning methods like Convolutional Neural Networks (CNNs) for improved feature extraction.

## Future Scope & Enhancements
🚀 Deep Learning Models: Implementing CNNs and Recurrent Neural Networks (RNNs) for more complex audio feature extraction.
🚀 Spotify API Integration: Analyzing real-time music trends and updating models dynamically.
🚀 Improved Feature Engineering: Exploring additional audio attributes like key, mode, and time signature for better predictions.
🚀 Expanding the Model: Training on a larger dataset covering global music trends across different regions and cultures.

By integrating advanced AI techniques, the accuracy of genre classification and popularity predictions can be significantly enhanced, contributing to smarter music recommendations for streaming platforms.

## Conclusion
This project successfully demonstrates how machine learning models can be used to classify songs into genres and predict their popularity scores based on audio attributes. By comparing multiple ML algorithms, we identified that Random Forest Classifier performs best for genre classification, while Random Forest Regressor provides the most accurate popularity predictions.

With further refinements and enhancements, this system could be incorporated into music streaming platforms to enhance user experiences by providing personalized recommendations and trend analysis.

