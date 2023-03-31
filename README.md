# Advanced_NLP_Project
In this project, we try to predict movie genres based on their description using an IMDB dataset from kaggle ([link](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)). We use a non-ML baseline model, an LSTM model, and a BERT model for the classification task. We find that our baseline term frequency model achieves a weighted-F-1 score of 0.410, our LSTM model a score of 0.642, and our BERT model a score of 0.653. In the notebook we take a closer look at the weaknesses of each model, at possible sources of bias in our models, and possible future improvements. 

Please note the following important contents of the repo:
* <tt>`final-advanced.ipynb`</tt> is the notebook that walks through the analysis.
* <tt>`helper_functions.py`</tt> includes some of the custom functions we wrote that were left out of the the notebook.
* <tt>`Genre Classification Dataset`</tt> contains the dataset.
* <tt>`test_predictions.csv`</tt> is the test dataset with the addition of all of the predictions of our different models except BERT.
* <tt>`test_predictions_with_bert.csv`</tt> is the test dataset with all of the predictions of our different models.

Nikita Baklazhenko, Miguel Conner, David Vallmanya, Dominik Wielath


