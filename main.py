import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords




#would like to write functions for the data cleaning so i can just have it all in one cell

# outputs a list of 'stop words' which are just words that arent of interest to the recommendation
stop_words = stopwords.words('english')

# create an instance of the class for the term-frequency inverse document frequency class
vectorizer = TfidfVectorizer(
    stop_words=stop_words, 
    max_df = 0.8,
    min_df = 5,
    ngram_range=(1,3),
    use_idf=True
    )

# takes in the anime csv file and cleans it ready for use
def clean_data():
    # reading in data
    df = pd.read_csv("C:/Users/isule/Downloads/animes.csv")
    #remove any shows that are scored below 7
    df.drop(df[df['score'] < 7].index, inplace = True)
    #picking the relevant columns
    anime_synopsises = df[['title', 'synopsis']]
    #drop rows with null values
    anime_synopsises_2 = anime_synopsises.dropna(how='any')
    #dropping any duplicate rows
    anime_synopsises_2 = anime_synopsises_2.drop_duplicates()
    #lower case
    anime_synopsises_2['synopsis'] = anime_synopsises_2['synopsis'].str.lower()
    # remove punctuation
    anime_synopsises_2['synopsis'] = anime_synopsises_2['synopsis'].str.replace(r'[^\w\s]+', '')
    # reset index after cleaning completed
    anime_synopsises_2 = anime_synopsises_2.reset_index(drop=True)

    return anime_synopsises_2


def get_reccomendation(synopsis, title):
    
    #use clean data function to get our cleaned data
    anime_synopsises_2 = clean_data()
     #lower case the input synopsis
    synopsis = synopsis.lower()
    # remove punctuation from input synopsis
    synopsis = synopsis.replace(r'[^\w\s]+', '')
    #add input to dataset
    # first store the index for later use
    indx = len(anime_synopsises_2.index)
    #insert the user input into our dataset
    anime_synopsises_2.loc[indx] = [title, synopsis]
    
    tf_idf_2 =  vectorizer.fit_transform(anime_synopsises_2['synopsis'])
  
  
    cosine_similarities = linear_kernel(tf_idf_2[(indx):(indx+1)], tf_idf_2).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-20:-1]
    
    return anime_synopsises_2.iloc[related_docs_indices]
    
