# Spotify Music Analysis and Popularity Prediction
Analyzing music data and predicting popularity of songs in Spotify with Machine Learning.
- Jupyter Notebook and PPT

## Overview
Has the music streaming service influenced the creation of a song? We would like to use the Spotify dataset to understand how the song is composed and what feature is affecting the popularity of the song.
We will analyze multiple features of music, such as energy, acousticness, danceability, valence, explicit, and popularity. It can provide us with a deep understanding of the spectrum of elements and the trend of music across a wide period. Then, we will build predictive models using machine learning and data mining; one for predicting the explicit and another for predicting the popularity of the song. The model can be applied as a popularity predictor for music producers, songwriters, marketers, streaming companies, etc.

[Dataset]

 - Trend Analysis Scope: The original data set consists of 19 attributes with 174,389 rows of Spotify music from 1920 to 2021.  
 - Prediction Analysis Scope: The filtered dataset consists of 19 attributes with 42,371 rows of Spotify music from 2000 to 2021.

[Dataset Features]
- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. (1.0 represents high confidence the track is acoustic.)
- danceability: Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. (1.0 is most danceable.)
- duration_ms: The duration of the track in milliseconds.
- energy: Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.
- id: The Spotify ID for the track.
- instrumentalness: Predicts whether a track contains no vocals. (1.0 indicates the greater likelihood the track contains no vocal content.
- popularity: Song ratings of Spotify audience.
- explicit: Explicit = 1 track is one that has curse words or language or art that is sexual, violent, or offensive in nature.
- mode: Indicates the modality (major = 1 or minor = 0) of a track.
- artists: Artists of the tracks.
- name: Name of the songs.

## Index
* Overview
* Research Questions
* Dataset Description
  - Trend Analysis Scope & Prediction Analysis Scope
* Exploratory Data Analysis
  - Numerical data description, Data Visualization, Correlation between features, Artists recommendation
* Trend Analysis
* Hypothesis Testing
  - Hypothesis test Analysis in danceability, Hypothesis test Analysis in valence, Hypothesis test Analysis in key
* Logistic Regression Model for Explicit Prediction
* Data Mining Techniques for Popularity Prediction
  - Popularity value prediction using regression models: Linear Regression, Decision Tree Regressor, XGB Regressor
  - Popularity level prediction using classification models: Random Forest Classfier, Logistic Regression, KNN Classifier
* Conclusion

