# Advanced Methods in Natural Language Processing - Final Project
# Predicting Movie Genres from Movie Descriptions
# Nikita Baklazhenko, Miguel Conner, David Vallmanya, Dominik Wielath

# Helper functions that we created for our project

import re
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer


def get_year(s):
    """Extracts the year from a string

    Parameters
    ----------
    s : String
        The string from which the year is extracted

    Returns
    -------
    year : int
        The extracted year
    """
    match1 = re.search(r'\((\d+)\)', s)   # search for a substring that matches the pattern "(digits)" and capture the digits as a group
    match2 = re.search(r'\((\d+)\/', s)   # search for a substring that matches the pattern "(digits)" and capture the digits as a group
    if match1:
        year = int(match1.group(1))       # extract the digits from the group and convert them to an integer
    elif match2:
    	year = int(match2.group(1))
    else:
        year = 0
    return year


def remove_stopwords(text, stop_words):
    """Removes stopwords form text

    Parameters
    ----------
    text : String
        The text from which stopwords are removed
    stopwords : list
        The list of stopwords that are removed

    Returns
    -------
    return value: String
        The text without the stop words
    """
    words = text.split()
    clean_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(clean_words)


def create_tfidf_dict(df,data_column, name_column, no_words):
    """Creates dictionary with the no_words number of words that have the
    highest TF-IDF score per topic for the provided data 

    Parameters
    ----------
    df : pandas dataframe
        A data frame containing text data with one row per topic
    data_column : String
        A string indicating the relevant data column 
    name_column : String
        A string indicating the relevant name column 
    no_words : int
        The number of words that will be included in the dictionary for each topic

    Returns
    -------
    important_dict: dict
        dictionary with the words that have the highest tf-idf score per topic
    """

    data = df[data_column]
    names = df[name_column]

    # create TF-IDF DataFrame
    vectorizer = TfidfVectorizer(max_df = 0.9)
    X = vectorizer.fit_transform(data)
    tokens = vectorizer.get_feature_names_out()
    X = X.toarray()
    X = pd.DataFrame(X, columns=tokens)
    X = X.T

    X.columns = names
    # create dictionary with most important words per topic
    important_dict = {}
    for col in X.columns:
        most_important = X.sort_values(by=col, ascending=False).head(no_words).index
        important_dict[col] = list(most_important)

    return important_dict
  
  
def dictionary_predictions(df, data_column, important_dict, genre_priors):
    """Predicts based on an dictionary which topic each text belongs to. 

    Parameters
    ----------
    df : pandas data frame
        A data frame containing text data
    data_columns : String
        The name of the column with the relevant data
    important_dict: dict
        A dictionary with the words that are relevant for the classification
    genre_priors: pandas data frame
        A data frame with the genres and their share in the training set

    Returns
    -------
    prediction: list
        contains a int value representing the topic predicted for each text 
    """

    data = df[data_column]

    # Initializing the list
    predictions = []
    i = 0
    # Iterrating over the texts (rows)
    for row in data:

        # Splitting the text in words
        words = row.split()

        # Initializing a dictionary to count how many words associated with a topic occur in each text
        count_dict = {}

        # Itterating over the topics in the important_dict dictionary
        for key in list(important_dict.keys()):
            top_words = set(important_dict[key])
            score = len(set(words) & top_words)
            count_dict[key] = score

        # Check if there is one topic with the largest number of occurences and make this the prediction    
        if sum(value == max(count_dict.values()) for value in count_dict.values()) == 1 :
            i += 1
            predictions.append(list(count_dict.keys())[list(count_dict.values()).index(max(count_dict.values()))])

        # Otherwise, make the prediction random based on the distribution of topics within the original data  
        else:
            predictions.append(np.random.choice(genre_priors.genre.values, size = 1, p = genre_priors.probability.values)[0])

    print("Share of data which was labeled based on dictionary: " + str(round(i/len(predictions), 3)))
    print("Share of data randomly labled using prior distribution: " + str(round(1 - i/len(predictions), 3)))
    return list(predictions)
