# Advanced Methods in Natural Language Processing - Final Project
# Predicting Movie Genres from Movie Descriptions
# Nikita Baklazhenko, Miguel Conner, David Vallmanya, Dominik Wielath

# Helper functions that we created for our project

import re
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
import torch
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def tokenize_and_preprocess(df, tokenizer, labels, max_length=512):
    """
    Tokenize and preprocess the text and labels from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text and labels.
        tokenizer (BertTokenizer): Tokenizer to use for tokenizing the text.
        labels (dict): Mapping of label names to label indices.
        max_length (int, optional): Maximum sequence length. Defaults to 512.

    Returns:
        tuple: Tuple containing tokenized texts and label indices.
    """
    texts = df['description'].tolist()
    encoded_texts = tokenizer(texts, padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
    labels1 = [labels[label] for label in df['genre']]
    return encoded_texts, labels1


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts['input_ids'][idx], self.texts['attention_mask'][idx], self.labels[idx]
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts['input_ids'][idx], self.texts['attention_mask'][idx], self.labels[idx]


class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer




def train(model, train_data, val_data, learning_rate, epochs):
    encoded_train_texts, train_labels = tokenize_and_preprocess(train_data)
    encoded_val_texts, val_labels = tokenize_and_preprocess(val_data)

    train_dataset = Dataset(encoded_train_texts, train_labels)
    val_dataset = Dataset(encoded_val_texts, val_labels)

    BATCH_SIZE = 1  # Increase the batch size if your GPU can handle it
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_mask, train_label in tqdm(train_dataloader):
            train_input, train_mask, train_label = train_input.to(device), train_mask.to(device), train_label.to(device)
            
            with autocast():  # Use mixed precision training
                output = model(train_input, train_mask)
                batch_loss = criterion(output, train_label.long())
            
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            with autocast():
                scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_mask, val_label in val_dataloader:
                val_input, val_mask, val_label = val_input.to(device), val_mask.to(device), val_label.to(device)
                output = model(val_input, val_mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        return model
