# Importing data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

#IMPORTING DATASET
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 44)
pd.set_option('display.width', 10000)

# Import a CSV file into a Pandas DataFrame
df = pd.read_csv("C:/Users/laura/Documents/Data_Analytics_Marketing/Project/MoviesOnStreamingPlatforms.csv")
print(df.head(10))
print(df.info())
print(df.describe())

#DATA CLEANING
#Drop Rotten Tomatoes,Unnamed: 0  column
df = df.drop(['Rotten Tomatoes', 'Unnamed: 0'], axis=1)

# Replacing missing values or dropping duplicates
# Fill Nan of Directors Column
df['Directors'].fillna("Unknown Director", inplace=True)
# print(df['Directors'].value_counts())

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="all")
df["Age"] = imputer.fit_transform(df[["Age"]]).ravel()

df.isna().sum()

df['Language'].fillna("Other", inplace=True)
# print(df['Language'].value_counts())

df['Runtime'].fillna(df['Runtime'].median(), inplace=True)


# Drop Nan of IMDb Column
Null = df.isnull().sum()
print(Null)
df = df.dropna(subset=['IMDb', 'Genres', 'Country'], axis=0)
Null_2 = (df.isnull().sum())
print(Null_2)
print(df.shape)

# Analyzing data
# Your project should use Regex to extract a pattern in data
C = df[df['Directors'].str.count(r'(^C.*)') > 0]

# total Directors starting with C
print('There are {} directors whose names start with the letter C'.format(C['Directors'].count()))

# Make use of Iterators
#Number of films for each director
df['Directors'][:100]
print(df['Directors'].value_counts()[:20])

obj_cols = df.dtypes[(df.dtypes == 'object')].index
print(obj_cols)
print(df[obj_cols].describe())

# Python
# Use functions to create reusable code
# Unique values for Genre by splitting the data every time a coma appears in the Genre column
def splitting(dataframe, col):
    split = dataframe[col].str.get_dummies(',')
    print('Movies Genres OK!')
    return split

genres = splitting(df, 'Genres')
print(genres)
# Merge dataframes
df = pd.concat([df, genres], axis=1, sort=False)
print(df.columns)

# Use functions from Numpy or Scipy

# # Use a Dictionary or Lists to store Data
# genre_list = list(df['Genres'].dropna().str.strip().str.split(","))
# flat_genre_list = []
# for sublist in genre_list:
#     for item in sublist:
#         flat_genre_list.append(item)
# print(set(flat_genre_list))

# Machine Learning
# Perform predictions using Supervised learning:
# IMDb Rating Prediction from a data set of Movies: https://www.kaggle.com/diptaraj23/imdb-rating-prediction-from-a-data-set-of-movies


# Perform hyper parameter tuning or Boosting whichever is relevant to your model.
# Your analysis should be relevant to marketing such as:
# customer behavioural analytics or segmentation

# Movies on OTT PLatforms - EDA and Clustering: https://www.kaggle.com/pranaykankariya/movies-on-ott-platforms-eda-and-clustering

# Select the features on the basis of which you want to cluster


# features = df[['Action', 'Adventure', 'Animation',
#                  'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
#                  'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',
#                  'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
#                  'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']].astype(int)
#
# #Scaling the data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(features)
#
# #Using TSNE
# tsne = TSNE(n_components=2)
# transformed_genre = tsne.fit_transform(scaled_data)
#
# #KMeans - Elbow Method
# distortions = []
# K = range(1,100)
# for k in K:
#     kmean = KMeans(n_clusters=k)
#     kmean.fit(scaled_data)
#     distortions.append(kmean.inertia_)
# fig = px.line(x=K,y=distortions,title='The Elbow Method Showing The Optimal K',
#               labels={'x':'No of Clusters','y':'Distortions'})
# fig.show()
#
#
# Kmeans
# cluster = KMeans(n_clusters=21)
# group_pred = cluster.fit_predict(scaled_data)
#
# tsne_df = pd.DataFrame(np.column_stack((transformed_genre,group_pred,df['Title'],df['Genres'])),columns=['X','Y','Group','Title','Genres'])
#
# fig = px.scatter(tsne_df,x='X',y='Y',hover_data=['Title','Genres'],color='Group',
#                  color_discrete_sequence=px.colors.cyclical.IceFire)
# fig.show()
#
#
#
# #Platforms with IMDB 8+ Movies
# netflix_films = df.loc[df['Netflix'] == 1].drop(['Hulu', 'Prime Video', 'Disney+', 'Type'],axis=1)
# hulu_films = df.loc[df['Hulu'] == 1].drop(['Netflix', 'Prime Video', 'Disney+', 'Type'],axis=1)
# prime_films = df.loc[df['Prime Video'] == 1].drop(['Netflix','Hulu', 'Disney+', 'Type'],axis=1)
# disney_films = df.loc[df['Disney+'] == 1].drop(['Netflix','Hulu', 'Prime Video', 'Type'],axis=1)
#
#
# count_imdb = [len(netflix_films[netflix_films['IMDb']>8]),len(hulu_films[hulu_films['IMDb']>8]),
#               len(prime_films[prime_films['IMDb']>8]),len(disney_films[disney_films['IMDb']>8])]
# platform = ['Netflix','Hulu','Prime Video','Disney+']
#
# top_rated = pd.DataFrame({'Platforms':platform,'Count':count_imdb})
# fig = px.bar(top_rated,x='Platforms',y='Count',color='Count',color_continuous_scale='Sunsetdark',title='IMDB 8+ Movies on different Platforms')
# fig.show()
#
# #Top Movies on Different Platforms
# net = netflix_films.sort_values('IMDb',ascending=False).head(10)
# hulu = hulu_films.sort_values('IMDb',ascending=False).head(10)
# prime = prime_films.sort_values('IMDb',ascending=False).head(10)
# disn = disney_films.sort_values('IMDb',ascending=False).head(10)
#
# fig = make_subplots(rows=4, cols=1,subplot_titles=("Top 10 Movies on Netflix","Top 10 Movies on Hulu",
#                                                    "Top 10 Movies on Prime Video","Top 10 Movies on Disney"))
#
# fig.add_trace(go.Bar(y=net['Title'],x=net['IMDb'],orientation='h',marker=dict(color=net['IMDb'],coloraxis="coloraxis"))
#              ,row=1,col=1)
# fig.add_trace(go.Bar(y=hulu['Title'],x=hulu['IMDb'],orientation='h',marker=dict(color=hulu['IMDb'], coloraxis="coloraxis")),row=2,col=1)
# fig.add_trace(go.Bar(y=prime['Title'],x=prime['IMDb'],orientation='h',marker=dict(color=prime['IMDb'], coloraxis="coloraxis")),row=3,col=1)
# fig.add_trace(go.Bar(y=disn['Title'],x=disn['IMDb'],orientation='h',marker=dict(color=disn['IMDb'], coloraxis="coloraxis")),row=4,col=1)
#
# fig.update_layout(height=1300, width=1000, title_text="Top Movies on Different Platforms based on IMDB Rating",
#                   coloraxis=dict(colorscale='Sunsetdark'),showlegend=False)
# fig.show()

# recommendation for market basket analysis
# OTT movies Recommendation in various platform: https://www.kaggle.com/karthikeyansh55/eda-movie-recommendation-using-plotly
# pandas_profiling.ProfileReport(df)

# Movies based by IMDB rating
# def round_val(data):
#     if int(data) != 'NaN':
#         return round(data)
#
# df['IMDB'] = df['IMDb'].apply(round_val)
#
# values = df['IMDB'].value_counts().sort_index(ascending=True).tolist()
# index = df['IMDB'].value_counts().sort_index(ascending=True).index
#
# values,index
#
# fig = px.bar(x=index, y=values, height = 400, color = index,
#             labels = { 'x' : 'IMDB rating', 'y' : 'Number of movies'})
# fig.show()
#
# #All platforms -Number of movies
# def val_sum(r,c):
#     return df[c].sum(axis=0)
#
# df_counts = []
# row = [df]
# col = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
#
# for x in row:
#     for y in col:
#         df_counts.append(val_sum(x,y))
#
#
# labels = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
#
# fig = go.Figure(data=[go.Pie(labels=labels, values=df_counts, hole=.5)])
# fig.update_layout(title_text = 'All platforms -Number of movies')
# fig.show()
#
# #
# def splitting(dataframe, col):
#     result = dataframe[col].str.get_dummies(',')
#     return result
#
# unique_genres = splitting(df, 'Genres')
# unique_lang = splitting(df, 'Language')
#
# newmovies = pd.concat([df, unique_genres], axis = 1)
#
#
# def val_sum(r,c):
#     return unique_genres[c].sum(axis=0)
#
# unique_counts = []
# row = [unique_genres]
# col = [unique_genres.columns]
#
# for x in row:
#     for y in col:
#         unique_counts.append(val_sum(x,y))
#
# plt.figure(figsize = (20, 10))
# unique_genres.sum().plot(kind="bar")
# plt.ylabel('Genres')
# plt.xlabel('Movies - Total Number')
# plt.title('Movies - Genres')
# plt.show()
#
#
# #Movies by year
# movies_by_year = newmovies.groupby('Year')['Title'].count().reset_index().rename(columns = {
#     'Title' : 'Movies - Total Number'
#      })
#
# fig = px.bar(movies_by_year, y = 'Year', x = 'Movies - Total Number', color = 'Movies - Total Number', orientation = 'h',
#              title = '1900 to 2020 total number of movies')
# fig.show()
# Visualize
# Top 10 Movies Netflix

# top_20 = df.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(20)
# fig = px.bar(top_20, x='Country', y='Number_of_Movies', color='Number_of_Movies', height=700,
#             title = 'Total number of movies based on country')
# fig.show()

# # Checking Number of Movies in a given Age group per Streaming Service
# top_20 = df.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(20)
# fig = px.bar(top_20, x='Country', y='Number_of_Movies', color='Number_of_Movies', height=700,
#             title = 'Total number of movies based on country')
# fig.show()
# print(df.head(10))

#MACHINE LEARNING
# Boosting
df_b = df.copy()
label_encoder = LabelEncoder()
df_b['Directors'] = label_encoder.fit_transform(df['Directors'])
df_b['Genres'] = label_encoder.fit_transform(df['Genres'])
df_b['Country'] = label_encoder.fit_transform(df['Country'])
df_b['Language'] = label_encoder.fit_transform(df['Language'])
df_b['IMDb'] = df_b['IMDb'].astype('int')
print(df_b.head(10))
df_xgboost = df_b[['Directors', 'Genres', 'Country', 'Language','Year', 'Runtime', 'IMDb',
                  'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime',
                  'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'Game-Show',
                  'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV',
                  'Romance', 'Sci-Fi', 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']]

x = df_xgboost.drop(labels='IMDb', axis=1)
y = df_xgboost['IMDb']
scaler=StandardScaler()
scaled_data=scaler.fit_transform(x)
train_x,test_x,train_y,test_y=train_test_split(scaled_data,y,test_size=0.3,random_state=42)
# fit model no training data
model = XGBClassifier(objective='multi:softmax', num_class=11)
model.fit(train_x, train_y)
# cheking training accuracy
y_pred = model.predict(train_x)
predictions = [value for value in y_pred]
accuracy_1 = accuracy_score(train_y,predictions)
print(accuracy_1)
# cheking initial test accuracy
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
accuracy_2 = accuracy_score(test_y,predictions)
print(accuracy_2)
print(test_x[0])
param_grid = {

    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001],
    'max_depth': [3, 5, 10, 20],
    'n_estimators': [10, 50, 100, 200]

}
grid= GridSearchCV(XGBClassifier(objective='multi:softmax', num_class=11),param_grid, verbose=3)
grid.fit(train_x,train_y)
print(grid.best_params_)

#Now to increase the accuracy of the model, we'll do hyperparameter tuning using grid search

#Decision Tree Classifier
# X_ran = df.drop(columns = 'IMDb')
# y_ran = df['IMDb']
#
# X_train, X_test, y_train, y_test = train_test_split(X_ran, y_ran, test_size = 0.20, random_state= 42)
#
# clf = DecisionTreeClassifier(min_samples_split= 2)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))


# clf2 = DecisionTreeClassifier(criterion = 'entropy', max_depth =24, min_samples_leaf= 1)
# clf2.fit(X_train,y_train)
# clf2.score(X_test,y_test)
# rand_clf = RandomForestClassifier(random_state=6)
# rand_clf.fit(X_train,y_train)
# print(rand_clf.score(X_test,y_test))
#
# grid_param = {
#     "n_estimators" : [90,100,115,130],
#     'criterion': ['gini', 'entropy'],
#     'max_depth' : range(2,20,1),
#     'min_samples_leaf' : range(1,10,1),
#     'min_samples_split': range(2,10,1),
#     'max_features' : ['auto','log2']
# }
# grid_search = GridSearchCV(estimator=rand_clf,param_grid=grid_param,cv=5,n_jobs =-1,verbose = 3)
# grid_search.fit(X_train,y_train)
#
# #let's see the best parameters as per our grid search
# print(grid_search.best_params_)

# Choose target and features
# print(df.isna().sum())
#
#
# df_n = df.copy()
# label_encoder = LabelEncoder()
# df_n['Directors'] = label_encoder.fit_transform(df['Directors'])
# df_n['Genres'] = label_encoder.fit_transform(df['Genres'])
# df_n['Country'] = label_encoder.fit_transform(df['Country'])
# df_n['Language'] = label_encoder.fit_transform(df['Language'])
# df_n['IMDb'] = df_n['IMDb'].astype('int')
# print(df_n.head(10))
#
# df_decision = df_n[['Directors', 'Genres', 'Country', 'Language','Year', 'Runtime', 'IMDb']]
#
#
# print(df_decision.info(5))
#
# X = df_decision.drop(columns='IMDb')
# y = df_decision['IMDb']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
# clf = DecisionTreeClassifier(min_samples_split=2)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
#
# clf2 = DecisionTreeClassifier(criterion ='entropy',max_depth=7,min_samples_leaf=1)
# clf2.fit(X_train, y_train)
# print(clf2.score(X_test, y_test))
# #Define the models
# rand_clf = RandomForestClassifier(criterion = 'gini',max_depth=10,random_state=42)
# rand_clf.fit(X_train,y_train)
# print(rand_clf.score(X_test,y_test))
#first test:
# grid_param = {
#     'n_estimators' : [10, 20, 50],
#     'criterion' : ['gini', 'entropy'],
#     'max_depth' : range(2,8,1),
#     'min_samples_split' : [2,4,5],
#     'max_features' : ['auto','log2']
# }
#Best Params: {'criterion': 'gini', 'max_depth': 7, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 50}

#Second test
# grid_param = {
#     'n_estimators' : [20, 50, 100],
#     'criterion' : ['gini', 'entropy'],
#     'max_depth' : range(2,8,1),
#     'min_samples_split' : range(2,5,1),
#     'min_samples_leaf' : range (1,5,1),
#     'max_features' : ['auto','log2']
# }
# grid_search = GridSearchCV(estimator=rand_clf, param_grid=grid_param,cv=5,n_jobs=-1,verbose=3)
# grid_search.fit(X_train,y_train)
# print(grid_search.best_params_)


#We will pass these parameters into our random forest classifier
# rand_clf = RandomForestClassifier(criterion= 'entropy',
#  max_depth = 7,
#  max_features = 'auto',
#  min_samples_leaf = 4,
#  min_samples_split= 2,
#  n_estimators = 100,random_state=6)
#
# rand_clf.fit(X_train, y_train)
# print(rand_clf.score(X_test, y_test))


# #Split the data for train and test
# X_train_full, X_test_full, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
# #List of Categorical colunmns to be used as features
# chosen_columns_cat = ['Directors', 'Genres', 'Country', 'Language']
# #List of Numerical colunmns to be used as features
# chosen_columns_num = ['Year', 'Runtime']
#
# #Keep selected columns only
# all_chosen_columns = chosen_columns_cat + chosen_columns_num
# X_train = X_train_full[all_chosen_columns]
# X_test = X_test_full[all_chosen_columns]
#
# print(X_train.head())
# print(X_test.head())
#
# #Copying the data to prevent change in original datset
# label_X_train = X_train.copy()
# label_X_test = X_test.copy()label_encoder.transform
#
# # Apply label encoder to each column with categorical data
# label_encoder = LabelEncoder()
# for col in chosen_columns_cat:
#     label_encoder.fit(pd.concat([label_X_train[col], label_X_test[col]], axis=0, sort=False))
#     label_X_train[col] = (label_X_train[col])
#     label_X_test[col] = label_encoder.transform(label_X_test[col])
#
# print(label_X_test[col].head(10))



# model_1 = RandomForestRegressor(n_estimators=50, random_state=1)
# model_2 = RandomForestRegressor(n_estimators=100, random_state=1)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=1)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=1)

# #List of models
# models = [model_1, model_2, model_3, model_4, model_5]
#
#
# def score_model(model, X_t=label_X_train, X_v=label_X_test, y_t=y_train, y_v=y_test):
#     model.fit(X_t, y_t)
#     preds = model.predict(X_v)
#     return mean_absolute_error(y_v, preds)
#
# mae_scores = []
#
# for i in range(0, len(models)):
#     mae = score_model(models[i])
#     print("Model %d MAE: %f" % (i + 1, mae))
#     mae_scores.append(mae)
#
# best_score=min(mae_scores)
# print(best_score)