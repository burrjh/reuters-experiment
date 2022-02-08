import pandas as pd
import numpy as np

u_cols = ['user_id', 'sex', 'age', 'occupation', 'zip_code']
users = pd.read_csv('users.dat', sep='::', names=u_cols, encoding='latin-1', engine='python')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')


#temporary df
cols = ['user_id', 'movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5']
df = pd.DataFrame(columns=cols)

df['user_id'] = ratings['user_id']
zeros = np.zeros(1000209)
df['movie_1']= zeros
df['movie_2']= zeros
df['movie_3']= zeros
df['movie_4']= zeros
df['movie_5']= zeros

for i in range(1000209):
    rating   = ratings.iloc[i,2]
    movie_id = ratings.iloc[i,1]
    
    df.iloc[i,rating] = movie_id  #note movie_id in cell corresponding to the integer of the rating


rating_dummies = pd.get_dummies(data=df, columns=['movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5'])

users_dummies = pd.dummies(data=users, columns=['sex', 'age', 'occupation'])


print(rating_dummies.head())


rating_dummies.to_csv("ratings_dummied")

users_dummies.to_csv("users_dummied")