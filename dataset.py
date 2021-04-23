# Using pytorch to implement an LSTM model on the text data, to predict whether the news is popular or not based on its title.
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NewsData:
    
    def __init__(self, data_path, votes_threshold):
        self.data_path = data_path
        self.votes_threshold = votes_threshold
        self.dataframe = self._process_data()
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Return features and label for the sample
        text = self.dataframe['title'].iloc[idx]
        up_votes = self.dataframe['up_votes'].iloc[idx]
        label = 1 if up_votes >= self.votes_threshold else 0
        return text, label
    
    def _get_class(self, up_votes):
        # Convert up_votes into one of the two classes: Popular or Not Popular
        return 1 if up_votes >= self.votes_threshold else 0
    
    def _process_data(self):
        # Read csv as pandas dataframe
        dataframe = pd.read_csv(self.data_path)
        # Drop down irrelevant features (Only "title" and "up_votes" are selected)
        dataframe = dataframe[['title', 'up_votes']]
        return dataframe
        
        