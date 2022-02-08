import pandas as pd 
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os 


dir_path = os.path.dirname(os.path.realpath(__file__))

os.chdir(dir_path)

u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
users = pd.read_table('users.dat', sep='::', names=u_cols,
                    encoding='latin-1', engine='python')

users = users.drop(['zip_code'], axis=1)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_table('ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')
ratings = ratings.drop(['unix_timestamp'], axis=1)

#temporary df
cols = ['user_id', 'movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5']
df = pd.DataFrame(columns=cols)

user_ids = list(set(ratings['user_id']))
zeros = np.zeros(len(user_ids))

df['user_id'] =  user_ids
df['movie_1'] = zeros
df['movie_2'] = zeros
df['movie_3'] = zeros
df['movie_4'] = zeros
df['movie_5'] = zeros


for id in range(len(user_ids)):
    
    dd = ratings[ratings['user_id']==id]

    for i in range(dd.shape[0]):
        rating = dd.iloc[i,2] # take the rating
        movie  = dd.iloc[i,1] # take the movie_id
        df.iloc[id-1,rating] = movie
    
#we now have a df with movie_<<rating>> as columns and the movie_id per user in the cells

# we now use pandas dummi function
dummied = pd.get_dummies(df, columns=['movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5'])


ratings_train, ratings_test = train_test_split(dummied, test_size=1200/6040, random_state=1111)

ratings_train.to_csv("ratings_train.csv")

ratings_test.to_csv("ratings_test.csv")

#the columns are named "movie_<rating>_<movie_id>"


#get dummies for users
users_dummies = pd.get_dummies(data=users, columns=['sex', 'age', 'occupation'])
users_dummies.to_csv("user_dummied.csv")

