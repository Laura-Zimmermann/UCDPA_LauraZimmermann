# Importing data
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import math
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from IPython.display import Image

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 17)
pd.set_option('display.width', 10000)

# Import a CSV file into a Pandas DataFrame
df = pd.read_csv("C:/Users/laura/Documents/Data_Analytics_Marketing/Project/MoviesOnStreamingPlatforms.csv")
print(df.head(10))
print(df.info())
print(df.describe())


# Analyzing data
# Your project should use Regex to extract a pattern in data
df['Directors'][:100]
print(df['Directors'].value_counts()[:20])

obj_cols = df.dtypes[(df.dtypes == 'object')].index
print(obj_cols)
print(df[obj_cols].describe())

# Replacing missing values or dropping duplicates
# Fill Nan of Directors Column
df['Directors'].fillna("Unknown Director", inplace=True)
#print(df['Directors'].value_counts())

df['Age'].fillna("Unknown Age", inplace=True)
#print(df['Age'].value_counts())

df['Rotten Tomatoes'].fillna("Unknown RT Score", inplace=True)
#print(df['Rotten Tomatoes'].value_counts())

df['Language'].fillna("Other", inplace=True)
#print(df['Language'].value_counts())

df['Runtime'].fillna("Unknown Runtime", inplace=True)
#print(df['Runtime'].value_counts())

# Drop Nan of IMDb Column
Null = df.isnull().sum()
print(Null)
df = df.dropna(subset=['IMDb', 'Genres', 'Country'], axis=0)
Null_2 = (df.isnull().sum())
print(Null_2)
print(df.shape)

# Make use of Iterators






# Merge dataframes
genres = df['Genres'].str.get_dummies(',')
df = pd.concat([df,genres],axis=1,sort=False)

# Python
# Use functions to create reusable code
# Unique values for Genre by splitting the data everytime a coma appears in the Genre column
def splitting(dataframe, col):
    split = dataframe[col].str.get_dummies(',')
    print('Movies Genres OK!')
    return split
unique_genres = splitting(df, 'Genres')
print(unique_genres)



print(df.columns)

# Use functions from Numpy or Scipy

# Use a Dictionary or Lists to store Data
genre_list = list(df['Genres'].dropna().str.strip().str.split(","))
flat_genre_list = []
for sublist in genre_list:
    for item in sublist:
        flat_genre_list.append(item)
print(set(flat_genre_list))

# Machine Learning
# Perform predictions using Supervised learning:
    # IMDb Rating Prediction from a data set of Movies: https://www.kaggle.com/diptaraj23/imdb-rating-prediction-from-a-data-set-of-movies







# Perform hyper parameter tuning or Boosting whichever is relevant to your model.
# Your analysis should be relevant to marketing such as:
    # customer behavioural analytics or segmentation

        #Movies on OTT PLatforms - EDA and Clustering: https://www.kaggle.com/pranaykankariya/movies-on-ott-platforms-eda-and-clustering

#Select the features on the basis of ehich you want to cluster


features = df[['Action', 'Adventure', 'Animation',
                 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                 'Fantasy', 'Film-Noir', 'Game-Show', 'History', 'Horror', 'Music',
                 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi',
                 'Short', 'Sport', 'Talk-Show', 'Thriller', 'War', 'Western']].astype(int)

#Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

#Using TSNE
tsne = TSNE(n_components=2)
transformed_genre = tsne.fit_transform(scaled_data)

#KMeans - Elbow Method
distortions = []
K = range(1,100)
for k in K:
    kmean = KMeans(n_clusters=k)
    kmean.fit(scaled_data)
    distortions.append(kmean.inertia_)
fig = px.line(x=K,y=distortions,title='The Elbow Method Showing The Optimal K',
              labels={'x':'No of Clusters','y':'Distortions'})
fig.show()

#We can observe that the “elbow” is the number 27 which is optimal for this case. We can also verify this beacuse there are 27 different genres so this result was pretty much expected.

#Kmeans
cluster = KMeans(n_clusters=21)
group_pred = cluster.fit_predict(scaled_data)

tsne_df = pd.DataFrame(np.column_stack((transformed_genre,group_pred,df['Title'],df['Genres'])),columns=['X','Y','Group','Title','Genres'])

fig = px.scatter(tsne_df,x='X',y='Y',hover_data=['Title','Genres'],color='Group',
                 color_discrete_sequence=px.colors.cyclical.IceFire)
fig.show()



#Platforms with IMDB 8+ Movies
netflix_films = df.loc[df['Netflix'] == 1].drop(['Hulu', 'Prime Video', 'Disney+', 'Type'],axis=1)
hulu_films = df.loc[df['Hulu'] == 1].drop(['Netflix', 'Prime Video', 'Disney+', 'Type'],axis=1)
prime_films = df.loc[df['Prime Video'] == 1].drop(['Netflix','Hulu', 'Disney+', 'Type'],axis=1)
disney_films = df.loc[df['Disney+'] == 1].drop(['Netflix','Hulu', 'Prime Video', 'Type'],axis=1)


count_imdb = [len(netflix_films[netflix_films['IMDb']>8]),len(hulu_films[hulu_films['IMDb']>8]),
              len(prime_films[prime_films['IMDb']>8]),len(disney_films[disney_films['IMDb']>8])]
platform = ['Netflix','Hulu','Prime Video','Disney+']

top_rated = pd.DataFrame({'Platforms':platform,'Count':count_imdb})
fig = px.bar(top_rated,x='Platforms',y='Count',color='Count',color_continuous_scale='Sunsetdark',title='IMDB 8+ Movies on different Platforms')
fig.show()

#Top Movies on Different Platforms
net = netflix_films.sort_values('IMDb',ascending=False).head(10)
hulu = hulu_films.sort_values('IMDb',ascending=False).head(10)
prime = prime_films.sort_values('IMDb',ascending=False).head(10)
disn = disney_films.sort_values('IMDb',ascending=False).head(10)

fig = make_subplots(rows=4, cols=1,subplot_titles=("Top 10 Movies on Netflix","Top 10 Movies on Hulu",
                                                   "Top 10 Movies on Prime Video","Top 10 Movies on Disney"))

fig.add_trace(go.Bar(y=net['Title'],x=net['IMDb'],orientation='h',marker=dict(color=net['IMDb'],coloraxis="coloraxis"))
             ,row=1,col=1)
fig.add_trace(go.Bar(y=hulu['Title'],x=hulu['IMDb'],orientation='h',marker=dict(color=hulu['IMDb'], coloraxis="coloraxis")),row=2,col=1)
fig.add_trace(go.Bar(y=prime['Title'],x=prime['IMDb'],orientation='h',marker=dict(color=prime['IMDb'], coloraxis="coloraxis")),row=3,col=1)
fig.add_trace(go.Bar(y=disn['Title'],x=disn['IMDb'],orientation='h',marker=dict(color=disn['IMDb'], coloraxis="coloraxis")),row=4,col=1)

fig.update_layout(height=1300, width=1000, title_text="Top Movies on Different Platforms based on IMDB Rating",
                  coloraxis=dict(colorscale='Sunsetdark'),showlegend=False)
fig.show()













    # recommendation for market basket analysis
        #OTT movies Recommendation in various platform: https://www.kaggle.com/karthikeyansh55/eda-movie-recommendation-using-plotly
#pandas_profiling.ProfileReport(df)

#Movies based by IMDB rating
def round_val(data):
    if int(data) != 'NaN':
        return round(data)

df['IMDB'] = df['IMDb'].apply(round_val)

values = df['IMDB'].value_counts().sort_index(ascending=True).tolist()
index = df['IMDB'].value_counts().sort_index(ascending=True).index

values,index

fig = px.bar(x=index, y=values, height = 400, color = index,
            labels = { 'x' : 'IMDB rating', 'y' : 'Number of movies'})
fig.show()

#All platforms -Number of movies
def val_sum(r,c):
    return df[c].sum(axis=0)

df_counts = []
row = [df]
col = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

for x in row:
    for y in col:
        df_counts.append(val_sum(x,y))


labels = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

fig = go.Figure(data=[go.Pie(labels=labels, values=df_counts, hole=.5)])
fig.update_layout(title_text = 'All platforms -Number of movies')
fig.show()

#
def splitting(dataframe, col):
    result = dataframe[col].str.get_dummies(',')
    return result

unique_genres = splitting(df, 'Genres')
unique_lang = splitting(df, 'Language')

newmovies = pd.concat([df, unique_genres], axis = 1)


def val_sum(r,c):
    return unique_genres[c].sum(axis=0)

unique_counts = []
row = [unique_genres]
col = [unique_genres.columns]

for x in row:
    for y in col:
        unique_counts.append(val_sum(x,y))

plt.figure(figsize = (20, 10))
unique_genres.sum().plot(kind="bar")
plt.ylabel('Genres')
plt.xlabel('Movies - Total Number')
plt.title('Movies - Genres')
plt.show()


#Movies by year
movies_by_year = newmovies.groupby('Year')['Title'].count().reset_index().rename(columns = {
    'Title' : 'Movies - Total Number'
     })

fig = px.bar(movies_by_year, y = 'Year', x = 'Movies - Total Number', color = 'Movies - Total Number', orientation = 'h',
             title = '1900 to 2020 total number of movies')
fig.show()
# Visualize
#Top 10 Movies Netflix

top_20 = df.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(20)
fig = px.bar(top_20, x='Country', y='Number_of_Movies', color='Number_of_Movies', height=700,
            title = 'Total number of movies based on country')
fig.show()

# Checking Number of Movies in a given Age group per Streaming Service
top_20 = df.groupby('Country')['Title'].count().reset_index().rename(columns = {'Title':'Number_of_Movies'}).sort_values('Number_of_Movies',ascending = False).head(20)
fig = px.bar(top_20, x='Country', y='Number_of_Movies', color='Number_of_Movies', height=700,
            title = 'Total number of movies based on country')
fig.show()
# Present two charts with Seaborn or Matplotlib
#X = df.drop(columns = 'IMDb')
#y = df['IMDb']
#chosen_columns_cat = ['Directors', 'Genres', 'Country', 'Language']
#chosen_columns_num = ['Year']
#all_chosen_columns = chosen_columns_cat + chosen_columns_num

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)

#clf = DecisionTreeClassifier()
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)

#y = df.IMDb
#X = df.drop(['IMDb'], axis=1)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#chosen_columns_cat = ['Directors', 'Genres', 'Country', 'Language']
#chosen_columns_num= ['Year', 'Runtime']
#all_chosen_columns = chosen_columns_cat + chosen_columns_num
#X_train = X_train[all_chosen_columns].copy()
#X_test = X_test[all_chosen_columns].copy()
#print(X_train.head())
#print(X_test.head())

#label_X_train = X_train.copy()
#label_X_test = X_test.copy()



#label_encoder = LabelEncoder()
#for col in chosen_columns_cat:
   # label_encoder.fit(pd.concat([label_X_train[col], label_X_test[col]], axis=0, sort=False))
    #label_X_train[col] = label_encoder.transform(label_X_train[col])
    #label_X_test[col] = label_encoder.transform(label_X_test[col])

#hip_1 = RandomForestRegressor(n_estimators=50, random_state=1)
#hip_2 = RandomForestRegressor(n_estimators=100, random_state=1)
#hip_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1)
#hip_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=1)
#hip_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=1)

#hip= [hip_1, hip_2, hip_3, hip_4, hip_5]

#def score_hip(hip, X_train, X_test, y_train, y_test):
    #hip.fit(X_train, y_train)
   # preds = hip.predict(X_test)
    #return mean_absolute_error(y_test, preds)

#mae_scores=[]

#for i in range(0, len(hip)):
    #mae = score_hip(hip[i])
    #print("Model %d MAE: %f" % (i+1, mae))
    #(mae_scores.append(mae)