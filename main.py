### IMPORTING DATA
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer



# IMPORTING DATASET
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 44)
pd.set_option('display.width', 10000)

# Import a CSV file into a Pandas DataFrame
df = pd.read_csv("dataset/MoviesOnStreamingPlatforms.csv")
print(df.head(10))
print(df.info())
print(df.describe())


### ANALYSING DATA
# DATA CLEANING

#Rotten Tomatoes Column is an object changing it to float
df['Rotten Tomatoes'] = df['Rotten Tomatoes'].str.rstrip('%').astype('float')
print(df['Rotten Tomatoes'])

# Replacing missing values or dropping duplicates
# Fill Nan of Directors Column
df['Directors'].fillna("Unknown Director", inplace=True)
print(df['Directors'].value_counts())

# Replace missing values of Age Column
# Use functions from Numpy or Scipy
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="all")
df["Age"] = imputer.fit_transform(df[["Age"]]).ravel()
df.isna().sum()

# Fill Nan of Language Column
df['Language'].fillna("Other", inplace=True)
print(df['Language'].value_counts())

# Replace Nan of Runtime Column with the median
df['Runtime'].fillna(df['Runtime'].median(), inplace=True)

# Drop Nan of IMDb Column
Null = df.isnull().sum()
print(Null)
df = df.dropna(subset=['IMDb', 'Genres', 'Country'], axis=0)
Null_2 = (df.isnull().sum())
print(Null_2)
print(df.shape)

# USING REGEX
# total of Directors starting with C
C = df[df['Directors'].str.count(r'(^C.*)') > 0]
print('There are {} directors whose names start with the letter C'.format(C['Directors'].count()))
# Finding all directors names that contain LlrR using Regex. Reason to choose this is because my initials are LR
regex = r"[^,]*[lLrR][^,]*(,[^,]*[lLrR][^,]*)*"

directors = df['Directors']
count = 0
for word in directors:
    if re.fullmatch(regex, word):
        count += 1

print(count)

# Number of films for each director
df['Directors'][:100]
print(df['Directors'].value_counts()[:20])

obj_cols = df.dtypes[(df.dtypes == 'object')].index
print(obj_cols)
print(df[obj_cols].describe())

df_directors = df.groupby('Directors')['Title'].count().reset_index().sort_values('Title', ascending=False).head(10)
print(df_directors.head())


### PYTHON
# Make use of Iterators
# Use functions to create reusable code
# Unique values for Genre by splitting the data every time a coma appears in the Genre column
def splitting(dataframe, col):
    split = dataframe[col].str.get_dummies(',')
    print(f'{col} OK!')
    return split

genres = splitting(df, 'Genres')
unique_lang = splitting(df, 'Language')
print(genres)

# Merge dataframes
df = pd.concat([df, genres], axis=1, sort=False)
print(df.columns)


# Use a Dictionary or Lists to store Data
genre_list = list(df['Genres'].dropna().str.strip().str.split(","))
flat_genre_list = []
for sublist in genre_list:
    for item in sublist:
        flat_genre_list.append(item)
genre_set = set(flat_genre_list)
count_genre = 0
for genre_item in flat_genre_list:
    if genre_item in genre_set:
        count_genre +=1
print(count_genre)

### VISUALIZATION, ANALYSIS
# Want to know the number of movies with IMDB score 7.5+ in each streaming platform
netflix_films = df.loc[df['Netflix'] == 1].drop(['Hulu', 'Prime Video', 'Disney+', 'Type'], axis=1)
hulu_films = df.loc[df['Hulu'] == 1].drop(['Netflix', 'Prime Video', 'Disney+', 'Type'], axis=1)
prime_films = df.loc[df['Prime Video'] == 1].drop(['Netflix', 'Hulu', 'Disney+', 'Type'], axis=1)
disney_films = df.loc[df['Disney+'] == 1].drop(['Netflix', 'Hulu', 'Prime Video', 'Type'], axis=1)

count_imdb = [len(netflix_films[netflix_films['IMDb'] > 7.5]), len(hulu_films[hulu_films['IMDb'] > 7.5]),
              len(prime_films[prime_films['IMDb'] > 7.5]), len(disney_films[disney_films['IMDb'] > 7.5])]

# Want to know the number of movies with Rotten Tomatoes score 75+ in each streaming platform
count_rotten = [len(netflix_films[netflix_films['Rotten Tomatoes'] > 75]), len(hulu_films[hulu_films['Rotten Tomatoes'] > 75]),
              len(prime_films[prime_films['Rotten Tomatoes'] > 75]), len(disney_films[disney_films['Rotten Tomatoes'] > 75])]
platform = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

top_rated_imdb = pd.DataFrame({'Platforms': platform, 'Count': count_imdb})
top_rated_imdb = top_rated_imdb.sort_values(['Count'], ascending=False)

fig_1 = px.bar(top_rated_imdb, x='Platforms', y='Count', color='Platforms', color_continuous_scale='rainbow',
               title='Number of IMDB Score 7.5+ Movies on each streaming platform', color_discrete_map= {'Netflix': 'red','Hulu': 'lawngreen', 'Prime Video': 'cornflowerblue','Disney+': 'darkblue'})
fig_1.show()

top_rated_rotten = pd.DataFrame({'Platforms': platform, 'Count': count_rotten})
top_rated_rotten = top_rated_rotten.sort_values(['Count'], ascending=False)

fig_2 = px.bar(top_rated_rotten, x='Platforms', y='Count', color='Platforms', color_continuous_scale='rainbow',
               title='Number of Rotten Tomatoes Score 75%+ Movies on each streaming platform', color_discrete_map= {'Netflix': 'red','Hulu': 'lawngreen', 'Prime Video': 'cornflowerblue','Disney+': 'darkblue'})
fig_2.show()

# Want to show top 5 Movies on each streaming platform
net = netflix_films.sort_values('IMDb', ascending=False).head(5)
hulu = hulu_films.sort_values('IMDb', ascending=False).head(5)
prime = prime_films.sort_values('IMDb', ascending=False).head(5)
disn = disney_films.sort_values('IMDb', ascending=False).head(5)

fig = make_subplots(rows=4, cols=1, subplot_titles=("Top 5 Movies on Netflix", "Top 5 Movies on Hulu",
                                                    "Top 5 Movies on Prime Video", "Top 5 Movies on Disney"))

fig.add_trace(go.Bar(y=net['Title'], x=net['IMDb'], orientation='h', marker=dict(color=net['IMDb'], coloraxis="coloraxis")), row=1, col=1)
fig.add_trace(go.Bar(y=hulu['Title'], x=hulu['IMDb'], orientation='h', marker=dict(color=hulu['IMDb'], coloraxis="coloraxis")),row=2, col=1)
fig.add_trace(go.Bar(y=prime['Title'], x=prime['IMDb'], orientation='h', marker=dict(color=prime['IMDb'], coloraxis="coloraxis")),row=3, col=1)
fig.add_trace(go.Bar(y=disn['Title'], x=disn['IMDb'], orientation='h', marker=dict(color=disn['IMDb'], coloraxis="coloraxis")),row=4, col=1)

fig.update_layout(height=1300, width=1000, title_text="Top Movies on Each Platform based on IMDB Score",
                  coloraxis=dict(colorscale='rainbow'), showlegend=False)
fig.show()


# Want to show the number of movies based by IMDB rating for each score points
def round_val(data):
    if int(data) != 'NaN':
        return round(data)


df['IMDB'] = df['IMDb'].apply(round_val)

values = df['IMDB'].value_counts().sort_index(ascending=True).tolist()
index = df['IMDB'].value_counts().sort_index(ascending=True).index

values, index

fig = px.bar(x=index, y=values, height=400, color=index,
             labels={'x': 'IMDB rating', 'y': 'Number of movies'})
fig.show()


# Want to show the percentage number of movies on each streaming platform
def val_sum(r, c):
    return df[c].sum(axis=0)


df_counts = []
row = [df]
col = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

for x in row:
    for y in col:
        df_counts.append(val_sum(x, y))

labels = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

fig = go.Figure(data=[go.Pie(labels=labels, values=df_counts, hole=.5)])
fig.update_layout(title_text='All platforms - Number of movies')
fig.show()

# Want to show the total number of movies by Genre
plt.figure(figsize=(20, 10))
genres.sum().sort_values(ascending=False).plot(kind="bar", color="magenta")
plt.ylabel('Genres')
plt.xlabel('Movies - Total Number')
plt.title('Movies - Genres')
# plt.show is blocking the rest of the code to run hence I have decided to comment it out otherwise the code won't run
# plt.show()

# Want to know the total number of Movies by year since 1980
year_count = df.groupby('Year')['Title'].count()
year_movie = df.groupby('Year')[['Netflix','Hulu','Prime Video','Disney+']].sum()
year_data = pd.concat([year_count,year_movie],axis=1).reset_index().rename(columns={'Title':'Movie Count'})
year_data_80 = year_data[year_data['Year']>=1980]

fig = px.bar(year_data_80,x='Year',y='Movie Count',hover_data=['Netflix','Hulu','Prime Video','Disney+'],
             color='Movie Count',color_continuous_scale='Sunsetdark',title='Movie Count By Year')
fig.show()


# Drop Rotten Tomatoes,Unnamed: 0  column, has too many Nan
df = df.drop(['Rotten Tomatoes', 'Unnamed: 0'], axis=1)


### MACHINE LEARNING
# Clustering
# Selecting the features I want to cluster
features = df[['Action', 'Adventure', 'Animation',
                 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',
                 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
                 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']].astype(int)

# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

# Using TSNE
tsne = TSNE(n_components=2)
genre_transform = tsne.fit_transform(scaled_data)

# KMeans -  The Elbow Method
distortions = []
K = range(1,100)

for k in K:
    kmean = KMeans(n_clusters=k, random_state=42, init='k-means++')
    kmean.fit(scaled_data)
    distortions.append(kmean.inertia_)
fig = px.line(x=K,y=distortions,title='Optimal K from The Elbow Method',
              labels={'x':'Number of Clusters','y':'Distortions'})
fig.show()

# Kmeans
cluster = KMeans(n_clusters=21, random_state=42, init='k-means++')
predict_group = cluster.fit_predict(scaled_data)

df_tsne = pd.DataFrame(np.column_stack((genre_transform, predict_group, df['Title'], df['Genres'])), columns=['X', 'Y', 'Group', 'Title', 'Genres'])

fig = px.scatter(df_tsne, x='X', y='Y', hover_data=['Title', 'Genres'], color='Group',
                 color_discrete_sequence=px.colors.cyclical.IceFire)
fig.show()



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

# Fitting the model in the training data
model = XGBClassifier(objective='multi:softmax', num_class=11)
model.fit(train_x, train_y)

# Cheking training accuracy
y_pred = model.predict(train_x)
predictions = [value for value in y_pred]
accuracy_1 = accuracy_score(train_y,predictions)
print(accuracy_1)

# Cheking initial test accuracy
y_pred = model.predict(test_x)
predictions = [round(value) for value in y_pred]
accuracy_2 = accuracy_score(test_y,predictions)
print(accuracy_2)
print(test_x[0])
param_grid = {

    'learning_rate': [1, 0.5, 0.1, 0.01],
    'max_depth': [3, 5, 10],
    'n_estimators': [10, 50, 100]

}
grid= GridSearchCV(XGBClassifier(objective='multi:softmax', num_class=11),param_grid, verbose=3)
grid.fit(train_x,train_y)
# To find the parameters giving maximum accuracy
print(grid.best_params_)

#Create new model using the same parameters
model_new=XGBClassifier(learning_rate= 0.1, max_depth= 3, n_estimators= 100)
model_new.fit(train_x, train_y)

y_pred_new = model_new.predict(test_x)
predictions_new = [round(value) for value in y_pred_new]
new_accuracy = accuracy_score(test_y, predictions_new)
print(new_accuracy)

# As we have increased the accuracy of the model, we'll save this model
filename = 'xgboost_model.pickle'
pickle.dump(model_new, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

# Saving the scaler object as well for prediction
filename_scaler = 'scaler_model.pickle'
pickle.dump(scaler, open(filename_scaler, 'wb'))

scaler_model = pickle.load(open(filename_scaler, 'rb'))

# Trying a random prediction
print(df_xgboost.head())

d = scaler_model.transform([[1893,173,1228,343,2010,136,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0]])
pred=loaded_model.predict(d)
print('This data belongs to class :',pred[0])


#Now to increase the accuracy of the model, we'll do hyperparameter tuning using grid search
#Decision Tree Classifier

df_tree = df_xgboost.copy()
X_ran = df_tree.drop(columns = 'IMDb')
y_ran = df_tree['IMDb']

X_train, X_test, y_train, y_test = train_test_split(X_ran, y_ran, test_size = 0.20, random_state= 42)

# Visualize the tree on the data without doing any pre processing
clf = DecisionTreeClassifier(min_samples_split= 2)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


clf2 = DecisionTreeClassifier(criterion = 'entropy', max_depth =24, min_samples_leaf= 1)
clf2.fit(X_train,y_train)
print(clf2.score(X_test,y_test))

# Using the RandomForestClassifier
rand_clf = RandomForestClassifier(random_state=6)
rand_clf.fit(X_train,y_train)
print(rand_clf.score(X_test,y_test))

#Tuning three hyperparameters, passing the different values for both parameters
grid_param = {
    "n_estimators" : [10,20,50],
    'criterion': ['entropy'],
    'max_depth' : range(2,5,1),
    'min_samples_leaf' : range(1,6,1),
    'min_samples_split': range(2,8,1),
    'max_features' : ['auto']
}
grid_search = GridSearchCV(estimator=rand_clf,param_grid=grid_param,cv=5,n_jobs =-1,verbose = 3)
grid_search.fit(X_train,y_train)

# Seeing the best parameters as per our grid search
print(grid_search.best_params_)

rand_clf = RandomForestClassifier(criterion= 'entropy',
 max_features = 'auto',
 max_depth = 4,
 min_samples_leaf = 1,
 min_samples_split= 2,
 n_estimators = 50,random_state=6)

rand_clf.fit(X_train,y_train)
print(rand_clf.score(X_test,y_test))


#Testing the Random Forest with different parameters and less columns:
df_n = df.copy()
label_encoder = LabelEncoder()
df_n['Directors'] = label_encoder.fit_transform(df['Directors'])
df_n['Genres'] = label_encoder.fit_transform(df['Genres'])
df_n['Country'] = label_encoder.fit_transform(df['Country'])
df_n['Language'] = label_encoder.fit_transform(df['Language'])
df_n['IMDb'] = df_n['IMDb'].astype('int')
print(df_n.head(10))

df_decision = df_n[['Directors', 'Genres', 'Country', 'Language','Year', 'Runtime', 'IMDb']]


print(df_decision.info(5))

X = df_decision.drop(columns='IMDb')
y = df_decision['IMDb']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

clf2 = DecisionTreeClassifier(criterion ='entropy',max_depth=7,min_samples_leaf=1)
clf2.fit(X_train, y_train)
print(clf2.score(X_test, y_test))
#Defining the models
rand_clf = RandomForestClassifier(criterion = 'gini',max_depth=10,random_state=42)
rand_clf.fit(X_train,y_train)
print(rand_clf.score(X_test,y_test))
##First test - I'm comenting this out as the results of the second test were better than the first test.
# grid_param = {
#     'n_estimators' : [10, 20, 50],
#     'criterion' : ['gini', 'entropy'],
#     'max_depth' : range(2,8,1),
#     'min_samples_split' : [2,4,5],
#     'max_features' : ['auto','log2']
# }

#These were the Best_Params printed: {'criterion': 'gini', 'max_depth': 7, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 50}

#Second test
grid_param = {
    'n_estimators' : [20, 50, 100],
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2,8,1),
    'min_samples_split' : range(2,5,1),
    'min_samples_leaf' : range (1,5,1),
    'max_features' : ['auto','log2']
}
grid_search = GridSearchCV(estimator=rand_clf, param_grid=grid_param,cv=5,n_jobs=-1,verbose=3)
grid_search.fit(X_train,y_train)
print(grid_search.best_params_)


#Passing these parameters into our random forest classifier
rand_clf = RandomForestClassifier(criterion= 'entropy',
 max_depth = 7,
 max_features = 'auto',
 min_samples_leaf = 4,
 min_samples_split= 2,
 n_estimators = 100,random_state=6)

rand_clf.fit(X_train, y_train)
print(rand_clf.score(X_test, y_test))
