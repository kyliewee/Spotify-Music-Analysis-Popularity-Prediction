# -*- coding: utf-8 -*-
"""
# Spotify Data Analysis and Popularity Prediction
# MS in Analytics, Northeastern University 
# Group: Gahyoung(Kylie) Lee, Yu-Chiao Shaw, Yichun Jin, Chia-Yun Chiang, Yuwei Hsu

#Introduction

#Dataset Description:
Numerical Features:
- acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. (1.0 represents high confidence the track is acoustic.)
- danceability: Describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. (1.0 is most danceable.)
- duration_ms: The duration of the track in milliseconds.
- energy: Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. 
- id: The Spotify ID for the track.
- instrumentalness: Predicts whether a track contains no vocals. (1.0 indicates the greater likelihood the track contains no vocal content. 
- key: The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.
- loudness: Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Values typical range between -60 and 0 decibels (db).
- release_date
- speechiness: Detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
- tempo: The overall estimated tempo of a track in beats per minute (BPM). Values typically range between 50 and 150.
- valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive, while tracks with low valence sound more negative.
- year

Target Feature:
- popularity: Song ratings of Spotify audience.

Dummy Features:
- explicit: Explicit = 1 track is one that has curse words or language or art that is sexual, violent or offensive in nature. 
- mode:  Indicates the modality (major = 1 or minor = 0) of a track.

Categorical Features:
- artists: Artists of the tracks.
- name: Name of the songs.

# Setup
"""

import warnings
warnings.filterwarnings("ignore")
 
import pydotplus
import requests
import io
import scipy
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_datareader as pdr
from pandas_datareader import data, wb
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import figure
from matplotlib.pyplot import style
import matplotlib.ticker as mtick
from datetime import date
from dateutil.parser import parse
from sklearn.feature_selection import RFE
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
import random
from numpy.random import randint
import matplotlib as mpl
from astropy.table import Table, Column
from tqdm import tqdm
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
from scipy.stats import chi2
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder

#time series
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set style
plt.style.use('seaborn-white')
plt.style.use('seaborn-pastel')
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (24,8)

path = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ13SeyXflqdDYhl9RQO4f45Eu-kPtQ9m-HFmotR48X30LoCRgaWx-Sc1M2hNDxg_A47poxhbylWInV/pub?gid=575812676&single=true&output=csv"
df_original = pd.read_csv(path)

df_original.shape

"""#Data Cleaning

"""

# Use only 2000-2020 subdataset
df= df_original[df_original.year >= 2000]

df.shape

# Drop duplicate records
df['artists+name'] = df.apply(lambda row: row['artists'] + row['name'], axis=1)
df_dup = df[df['artists+name'].duplicated()]
df_dup.shape

df['artists+name'].unique

df.columns

# Generate new dataframe which includes the needed columns for future use
df = df.groupby("artists+name").agg({"acousticness":"max", "danceability":"max",
                                "duration_ms":"max", "energy":"max",
                                "explicit":"max", "instrumentalness":"max",
                                "key":"max", "liveness":"max", "loudness":"max",
                                "mode":"max", "artists":"max", "popularity":"max", 
                                "speechiness":"max", "tempo":"max", "valence":"max",
                                "year":"max"})
df.shape

df.head(3)

df_artists = df.groupby(by =['artists']).mean().reset_index().sort_values(by='popularity', ascending=False)

df.reset_index(level=[0], inplace=True)

"""# EDA"""

df.head(3)

df.tail(3)

# Transform milliseconds to minutes
df["duration_mins"] = df["duration_ms"]/60000
df.drop(columns="duration_ms", inplace=True)

df.columns

# Drop extraneous columns
df.drop(columns=["artists+name", "artists"], inplace=True)

df.columns

df.info()

df.isnull().sum()

df.describe()

"""Summary:   
- tempo = 0 is not reasonable. Tempo typically ranges from 50 to 150, we would like to drop it.

"""

# Drop the records with tempo = 0
df = df[df.tempo != 0]

# Check the values of tempo after cleansing
df.tempo.value_counts()

"""# Trend Analysis"""

df.popularity.describe()

df_year = df_original.groupby(by=["year"]).mean().reset_index()
df_year

# Create a line plot to see the trends
plt.title("Song Trends Over Time", fontdict={"fontsize": 15})

lines = ["acousticness","danceability","energy", 
         "instrumentalness", "liveness", "valence", "speechiness"]

for line in lines:
    ax = sns.lineplot(x='year', y=line, data=df_year)
    
    
plt.ylabel("value")
plt.legend(lines,loc = 'upper right');

# Popularity trend
df_year = df_original.groupby(by=["year"]).popularity.mean().reset_index()
pop = df_year.set_index('year')
pop = pd.DataFrame(pop)
pop.plot(title = 'Popularity Over Time');

df_popularity = df_year[['year','popularity']]

# Convert the data type of year to Datetime
df_popularity['year'] = pd.to_datetime(df_popularity['year'], format = '%Y')

# Set the index as year
df_popularity.set_index('year',inplace = True)

df_popularity.head()

# Decomposition
pop_decomposition = sm.tsa.seasonal_decompose(df_popularity, model = 'additive')
fig = pop_decomposition.plot()
plt.title('Decomposition of Popularity')
plt.show()

# Define function for ADF test
def adf_test(timeseries):
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

# Apply adf test on the series
adf_test(df_popularity['popularity'])

# First order differencing
pop_diff = df_popularity['popularity'] - df_popularity['popularity'].shift(1)
pop_diff.dropna(axis='index', inplace=True)
pop_diff.plot(title = 'Differenced Popularity Over Year');

adf_test(pop_diff)

# Transformation
pop_log = np.log(df_popularity['popularity'])
pop_log_diff = df_popularity['popularity'] - df_popularity['popularity'].shift(1)
pop_log_diff.dropna(axis='index', inplace=True)
pop_log_diff.plot(title = 'Logged Popularity Over Time')

adf_test(pop_log_diff)

# acf and pacf

fig, ax = plt.subplots(2,figsize=(20,8))
ax[0] = plot_acf(df_popularity['popularity'], ax=ax[0], lags=100)
ax[1] = plot_pacf(df_popularity['popularity'], ax=ax[1], lags=50)

"""# Data Visualization"""

# Define function which generates distplot
def generate_distplot(df, target_column, compare_column, subset0_name, subset1_name):
  plt.figure()
  subset0 = df[compare_column].loc[df[target_column] == 0]
  subset1 = df[compare_column].loc[df[target_column] == 1]
  sns.set_style('whitegrid')
  ax = sns.distplot(subset0, hist=True, kde_kws=dict(linewidth=4), color = "lightsalmon" )
  ax = sns.distplot(subset1, hist=True, kde_kws=dict(linewidth=4), color = "lightseagreen")
  plt.legend([subset0_name, subset1_name],fontsize=20)

df.describe()

# Visualize histogram - popularity
ax = plt.hist(df['popularity'], bins=10)
for i in range(10):
    plt.text(ax[1][i] + 2.5,ax[0][i],str(ax[0][i]))
plt.xlabel('Popularity')
plt.ylabel('Song Numbers')
plt.title('Popularity Distribution');

# Visualize distplot - explicit
obj_list = list(df.select_dtypes(include = ["object"]).columns)
num_column = df.columns.difference(obj_list)
for i in num_column:
  generate_distplot(df, "explicit", i, "not explicit", "explicit")

# Visualize distplot - mode
for i in num_column:
  generate_distplot(df, "mode", i, "minor", "major")

# Check the correlation between features
corr = df.corr() 
f,ax = plt.subplots()
sns.heatmap(corr, annot = True, fmt= '.2f', 
            xticklabels= True, yticklabels= True
            ,cmap="coolwarm", linewidths=.5, ax=ax)
plt.title('Correlation heatmap', size=15);

# Data visualization of every feature
plt.style.use('fivethirtyeight')
cols = [ f for f in df.columns if df.dtypes[ f ] != "object"]
 
f = pd.melt( df, value_vars=cols)
g = sns.FacetGrid( f, col="variable", col_wrap=6, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()

# Distplot
plt.style.use('fivethirtyeight')
output = 'explicit'
 
cols = [ f for f in df.columns if df.dtypes[ f ] != "object"]
cols.remove( output )
 
f = pd.melt( df, id_vars=output, value_vars=cols)
g = sns.FacetGrid( f, hue=output, col="variable", col_wrap=6, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()

"""Summary   
- Songs containing explicit contents are moderately danceable , energetic and have higher valence. 
- Explicit songs tend to have higher popularity scores, and these songs released more in recent years.

### Features' range for popular songs
"""

df_popular = df[df['popularity'] > 75]
df_popular.shape

des = df_popular.describe().round(2)
des

# Visualizon
cols = [ f for f in df_popular.columns if df_popular.dtypes[ f ] != "object"]
 
f = pd.melt( df_popular, value_vars=cols)
g = sns.FacetGrid( f, col="variable", col_wrap=6, sharex=False, sharey=False )
g = g.map( sns.distplot, "value", kde=True).add_legend()

# Calculate the confidence intervals for 95%
f_mean = des.iloc[1]
f_std = des.iloc[2]
interval_min = (f_mean.values - 1.96 * f_std / np.sqrt(1274))
interval_max = (f_mean.values + 1.96 * f_std / np.sqrt(1274))
min = pd.DataFrame(interval_min)
min.rename(columns={"std": "min"}, inplace=True)
max = pd.DataFrame(interval_max)
max.rename(columns={"std": "max"}, inplace=True)
pd.concat([min, max],axis=1)
mean = pd.DataFrame(f_mean)
pd.concat([min, mean, max],axis=1).round(2)

"""### Artists recommendation"""

# Define function which shows top three artists from specific feature
def recomend_artists(feature):
  recomendation = df_artists[df_artists[feature] > df_artists[feature].quantile(0.9)].sort_values(by='popularity', ascending=False)
  return recomendation.head(3)

# Top 3 popluar artists:
df_artists.head(3)

# Top 3 Popluar artists with high danceability:
recomend_artists('danceability')

# Top 3 Popluar artists with high instrumentalness:
recomend_artists('instrumentalness')

"""# Hypothesis Testing"""

# ANOVA test - danceability
plt.style.use('fivethirtyeight')
explicit_data0 = df["danceability"].loc[df["explicit"] == 0]
explicit_data1 = df["danceability"].loc[df["explicit"] == 1]

ax = sns.distplot(explicit_data0, hist=True, kde_kws=dict(linewidth=4))
ax = sns.distplot(explicit_data1, hist=True, kde_kws=dict(linewidth=4))
plt.legend(['Explicit = 0','Explicit = 1'],fontsize=20)
plt.title('Density of Danceability\ncut by Explict')

# ANOVA - danceability
alpha = 0.05
explicit_stat, explicit_p = f_oneway(explicit_data0, explicit_data1)
print("Explicit statistics is %.3f, p value is %.3f" % (explicit_stat, explicit_p))
if explicit_p <= alpha:
  print("Reject H0. Different distribution (alpha = 0.05) ")
else:
  print("Fail to reject H0. Same distribution (alpha = 0.05)")

"""Hypothesis test Analysis in danceability:
1. Test Setup:
- The Null Hypothesis (H0): The danceability of the tracks without explicit lyrics is the same as those songs with explicit songs.
- The Alternate Hypothesis (H1): The danceability of the tracks without explicit lyrics is different than those songs with explicit songs.
- The alpha-value will be set to 0.05.
2. Test Analysis:
- Claim is H0.
- Since p-value = 0 and it is less than the alpha-value = 0.05. We reject the null hypothesis.
- There is enough evidence to reject the claim that the danceability is the same regardless of the explicit contents.
- The danceability distribution of explicit songs is different than the distribution of songs without explicit contents.


"""

# ANOVA test - valence
explicit_data0 = df["valence"].loc[df["explicit"] == 0]
explicit_data1 = df["valence"].loc[df["explicit"] == 1]

ax = sns.distplot(explicit_data0, hist=True, kde_kws=dict(linewidth=4))
ax = sns.distplot(explicit_data1, hist=True, kde_kws=dict(linewidth=4))
plt.legend(['Explicit = 0','Explicit = 1'],fontsize=20)
plt.title('Density of Valence\ncut by Explict')

# ANOVA - valence
alpha = 0.05
explicit_stat, explicit_p = f_oneway(explicit_data0, explicit_data1)
print("Explicit statistics is %.3f, p value is %.3f" % (explicit_stat, explicit_p))
if explicit_p <= alpha:
  print("Reject H0. Different distribution (alpha = 0.05) ")
else:
  print("Fail to reject H0. Same distribution (alpha = 0.05)")

"""Hypothesis test Analysis in valence:
1. Test Setup:
- The Null Hypothesis (H0): The valence of the tracks without explicit lyrics is the same as those songs with explicit songs.
- The Alternate Hypothesis (H1): The valence of the tracks without explicit lyrics is different than those songs with explicit songs.
- The alpha-value will be set to 0.05.
2. Test Analysis:
- Claim is H0.
- Since p-value = 0.559 and it is much higher than the alpha-value = 0.05. We fail to reject the null hypothesis.
- The valence distribution of explicit songs is the same as songs without explicit contents .

"""

# Chi-square independent test
df.groupby("key")["explicit"].value_counts()

# Test for independent using Pearsons's Chi-squared Test
# H0: two variables are independent; H1: two variables are dependent
test_data   = [[3673, 556],
               [2889, 999],
               [3262, 514],
               [959, 133],
               [2500, 398],
               [2566, 398],
               [2173, 465],
               [3878, 596],
               [1977, 482],
               [3274, 471],
               [2065, 449],
               [2519, 569]]
alpha = 0.05
stat, p, dof, expected = chi2_contingency(test_data)

if p <= alpha:
  print("Reject H0. Explicit is dependent with key. (alpha = 0.05)")
else:
  print("Fail to reject H0. Explicit is independent with key. (alpha = 0.05)")

"""Hypothesis test Analysis in key
1. Test Setup:
- The Null Hypothesis (H0): The key of songs is independent with explicit content .
- The Alternate Hypothesis (H1): The key of songs is dependent with explicit content.
- The alpha-value will be set to 0.05.
2. Test Analysis:
- Claim is H0.
- Since p-value is less than alpha value. We reject the null hypothesis.
- The key of songs is dependent with explicit content.

# Logistic Regression Model
"""

# Define function which find the best K (number of attributes) and the exact attributes as our training data
def run_logistic_regression_with_different_k(df, target_column):
    # Assign feature selection data 
    fs_y = df[target_column]
    fs_X = df.loc[:, df.columns != target_column]
    # Iterate with the number of X column 
    for i in range(fs_X.shape[1]):
      lr_model = LogisticRegression(max_iter=1000)
      rfe = RFE(lr_model, i+1)
      fit = rfe.fit(fs_X, fs_y)
      X_column = []
      # Iterate with fit.ranking to get the best attribute
      for j in range(len(fit.ranking_)):
        if fit.ranking_[j] == 1: # fit.ranking_ number equals to 1 indicates this attribute we want to select
          X_column.append(fs_X.columns[j])

      # Assign training and target data
      X = fs_X[X_column].values
      y = df[target_column]
      # Train and fit model
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
      lr_model.fit(X_train, y_train)
      y_predict = lr_model.predict(X_test)

      # Model evaluation metrics
      accuracy = accuracy_score(y_test, y_predict)
      mse = mean_squared_error(y_test, y_predict)
      cr_result = classification_report(y_test, y_predict)
      cv_scores = cross_val_score(lr_model, X, y, scoring='accuracy', cv=10)

      print("Select {} attributes as training data".format(i+1) )
      print("The training attributes includes:", X_column) 
      print("Model performance as below:")
      print("(1) Classification_report:")
      print(cr_result)
      print("(2) Accuracy:", accuracy) 
      print("(3) MSE:", mse)
      print("(4) Cross validation scores", cv_scores)
      print("(5) Cross validation mean scores", cv_scores.mean())
      print("----------------------------------------------------------")

run_logistic_regression_with_different_k(df, "explicit")

X = df[['acousticness', 'danceability', 'energy', 'instrumentalness',
        'key', 'liveness', 'loudness', 'mode', 'popularity', 
        'speechiness', 'tempo', 'valence', 'year', 'duration_mins']].values
y = df["explicit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
lr_explicit_model = LogisticRegression(max_iter=1000)
lr_explicit_model.fit(X_train, y_train)
y_predict = lr_explicit_model.predict(X_test)

# Model evaluation metrics
accuracy = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
cr_result = classification_report(y_test, y_predict)
cv_scores = cross_val_score(lr_explicit_model, X, y, scoring='accuracy', cv=10)

print("Model performance as below:")
print("(1) Classification_report:")
print(cr_result)
print("(2) Accuracy:", accuracy) 
print("(3) MSE:", mse)
print("(4) Cross validation scores", cv_scores)
print("(5) Cross validation mean scores", cv_scores.mean())

"""# Prediction Model for popularity

# Predict for specific value (continuous variable)
"""

def run_regression_model(model, alg_name):
   # build the model on training data
   model.fit(X_train, y_train)
   # make predictions for test data
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)  
   print("Model: ", alg_name)
   print("mse:", mse)
   print("Predictions", y_pred)

def run_model(model, alg_name):
   # build the model on training data
   model.fit(X_train, y_train)
 
   # make predictions for test data
   y_pred = model.predict(X_test)
   # calculate the accuracy score
   accuracy =  accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)
   scoresDT3 = cross_val_score(model, X_test, y_test, cv=6)
   Cr = classification_report(y_test, y_pred)
   
   print("Model: ", alg_name)
   print("Accuracy on Test Set for {} = {:.2f}\n".format(alg_name,accuracy))
   print(Cr)
   print("{}: CrossVal Accuracy Mean: {:.2f} and Standard Deviation: {:.2f} \n".format(alg_name,scoresDT3.mean(), scoresDT3.std()))

# Prepare data for prediction
y = df['popularity']
X = df.drop(columns=['popularity'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Transform features by scaling specific features to a given range
ctr = ColumnTransformer([('minmax', MinMaxScaler(), ['year', 'tempo', 'duration_mins']),
                        ('categorical', OneHotEncoder(), ['key'])],
                       remainder='passthrough')

ctr.fit(X_train)
X_train_preprocessed = ctr.transform(X_train)
X_test_preprocessed = ctr.transform(X_test)

# Linear Regression
model = LinearRegression()
run_regression_model(model, "Linear Regression")

# Decision Tree Regression
model = DecisionTreeRegressor()
run_regression_model(model, "Decision Tree Regressor")

# XGB Regression
model = XGBRegressor()
run_regression_model(model, "XGB Regressor")

# Predicted values visualization for XGB model
y_pred = model.predict(X_test)
x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=10, color="blue", label="original")
plt.scatter(x_ax, y_pred, s=11, color="red", label="predicted")
plt.title("Popularity test and predicted data")
plt.legend()
plt.show()

"""# Categorize the popularity into 4 levels"""

df_c = df.copy()

df_c.loc[((df.popularity >= 0) & (df.popularity <= 25)), "popularity_level" ] = 1
df_c.loc[((df.popularity > 25) & (df.popularity <= 50)), "popularity_level" ] = 2
df_c.loc[((df.popularity > 50) & (df.popularity <= 75)), "popularity_level" ] = 3
df_c.loc[((df.popularity > 75) & (df.popularity <= 100)), "popularity_level" ] = 4

df_c["popularity_level"] = df_c["popularity_level"].astype("int")

df_c.drop(columns="popularity", inplace=True)

df_c

y = df_c['popularity_level']
X = df_c.drop(columns=['popularity_level'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

ctr = ColumnTransformer([('minmax', MinMaxScaler(), ['year', 'tempo', 'duration_mins']),
                        ('categorical', OneHotEncoder(), ['key'])],
                       remainder='passthrough')

ctr .fit(X_train)
X_train_preprocessed = ctr .transform(X_train)
X_test_preprocessed = ctr .transform(X_test)

model = RandomForestClassifier(n_estimators=10)
run_model(model, "Random Forest")

model = LogisticRegression(multi_class='multinomial')
run_model(model, "Logistic Regression")

model = KNeighborsClassifier()
run_model(model, "Nearest Neighbors Classifier")

# Random Forest Model as our prediction model
rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy =  accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
scoresDT3 = cross_val_score(rf_model, X_test, y_test, cv=6)
Cr = classification_report(y_test, y_pred)
   
print("Model: Random Forest model")
print("Accuracy on Test Set ={}".format(accuracy))
print(Cr)
print("CrossVal Accuracy Mean: {:.2f} and Standard Deviation: {:.2f} \n".format(scoresDT3.mean(), scoresDT3.std()))

# Generate 1000 random sample
random_input = pd.DataFrame(columns = df_c.columns)
random.seed(42)
for i in range(1000):
  x = [random.uniform(0, 1) for i in range(3)]
  x.append(random.randint(0,1))
  x.append(random.uniform(0,1))
  x.append(random.randint(0,11))
  x.append(random.randint(0,1))
  x.append(random.randint(-9,-5))
  x.append(random.randint(0,1))
  x.append(random.uniform(0,1))
  x.append(random.uniform(100,150))
  x.append(random.uniform(0,1))
  x.append(random.randint(2000,2021))
  x.append(random.uniform(0,88))
  x.append(random.randint(0,4))
  random_input.loc[i] = x

sample_data = random_input.drop(columns=["popularity_level"])

sample_data

y_sample_pred = rf_model.predict(sample_data)

sample_data = sample_data.join(pd.DataFrame(y_sample_pred, columns= {"popularity_level"}))

sample_data

sample_data.popularity_level.value_counts()

sample_data.popularity_level

# Visualize histogram - predicted popularity
ax = plt.hist(sample_data['popularity_level'], bins=3)
for i in range(3):
    plt.text(ax[1][i] +0.3, ax[0][i],str(ax[0][i]))
plt.xlabel('Popularity')
plt.ylabel('Song Numbers')
plt.title('Popularity Distribution');

"""##"""
