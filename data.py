#!/usr/bin/python3

'''
Author: Ambareesh Ravi
Date: 27 July, 2021
File: data.py
Description:
    Handles the loading of dataset
'''

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    # Class to load data as csv file
    def __init__(self, file_name = "data/spam.csv"):
        '''
        Initializes the class object

        Args:
            file_name: file path as <str>
        Returns:
            -
        Exception:
            -
        '''
        self.df = pd.read_csv(file_name)
        self.df["v1"] = [1 if i == "spam" else 0 for i in self.df["v1"]]
        self.X = np.array(self.df["v2"])
        self.y = np.array(self.df["v1"])
    
    def __call__(self,):
        '''
        Object call returns the dataframe

        Args:
            -
        Returns:
            data and labels as as <tuple>
        Exception:
            -
        '''
        return (self.X, self.y)

    def balanced_sample(self, test_size = 0.25):
        '''
        Returns balanced sampling for training and testing the models

        Args:
            -
        Returns:
            train_data, train_labels, test_data, test_labels
        Exception:
            -
        '''
        normal_indices = np.argwhere(self.y == 0)
        spam_indices = np.argwhere(self.y == 1)

        train_spam, test_spam = train_test_split(spam_indices, test_size = test_size, shuffle = True)
        train_normal, test_normal = train_test_split(normal_indices, test_size = test_size, shuffle = True)

        train_indices = np.concatenate((train_normal, train_spam))
        test_indices = np.concatenate((test_normal, test_spam))

        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        return self.X[train_indices].squeeze(), self.y[train_indices].squeeze(), self.X[test_indices].squeeze(), self.y[test_indices].squeeze()