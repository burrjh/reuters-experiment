import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_table('ratings.dat', sep='::', names=r_cols, encoding='latin-1', engine='python')
ratings = ratings.drop(['unix_timestamp'], axis=1)


rates = list(set(ratings['rating']))
movie_ids = list(set(ratings['movie_id']))
user_ids  = list(set(ratings['user_id']))

col_names = ['user_id']

for movie in movie_ids:
    for rating in rates:
        col_names.append( "movie_"+str(movie)+"_"+str(rating))


df = pd.DataFrame(0, index=range(6040),columns=col_names) #init df with zeros
df['user_id']=user_ids


for user_id in user_ids:
    
    dd = ratings[ratings['user_id']==user_id] #temporary dataframe

    for i in range(dd.shape[0]):

        movie_id = dd.iloc[i,1] #first element
        rating   = dd.iloc[i,2] #second element

        col_name = "movie_"+str(movie_id)+"_"+str(rating)

        df.loc[user_id-1,col_name]=1




df.to_csv("dummy_ratings.csv")


df2 = df.loc[:, (df != 0).any(axis=0)]  #dropping columns with only-zero entries


#df2.to_csv("ratings_prep.csv")
    

ratings_train, ratings_test = train_test_split(df2, test_size=1200/6040, random_state=1111)

ratings_train.to_csv("ratings_train.csv")

ratings_test.to_csv("ratings_test.csv")



    