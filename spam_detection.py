#!/usr/bin/python3

'''
Author: Ambareesh Ravi
Date: 27 July, 2021
File: spam_detection.py
Description:
    trains, evaluates and predicts text spam detection model
'''
# imports
import os
import argparse
import pickle

# nltk imports
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# model imports
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# module imports
from data import *

class DenseTransformer(TransformerMixin):
    # Creates a dummy transformer to convert sparse matrices into dense
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

class SpamDetector:
    # Detects whether or not a text is a spam
    def __init__(self,):
        '''
        Initializes the class
        
        Args:
            -
        Returns:
            -
        Exception:
            -

        MultinomialNB
        RandomForestClassifier
        GradientBoostingClassifier
        XGBoost
        '''

        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        self.stemmer = PorterStemmer()
        
    def process_content(self, content):
        '''
        Preprocess the contents of a sentence
        
        Args:
            content as <str>
        Returns:
            <np.array> of preprocessed words
        Exception:
            -
        '''

        content = content.lower()
        content = self.regex_tokenizer.tokenize(content)
        # Ignore stop words
        stemmed_tokens = [self.stemmer.stem(wt) for wt in content if wt not in stopwords.words('english')]
        return np.array(stemmed_tokens)

    def form_pipeline(self,):
        '''
        creates the sklearn pipeline for preprocessing inputs to the model
        
        Args:
            -
        Returns:
            <sklearn.pipeline.Pipeline> object
        Exception:
            -
        '''
        self.pipeline = Pipeline([
            ('count_vec', CountVectorizer(analyzer = self.process_content)), 
            ('to_dense', DenseTransformer()),
            ('TF_IDF', TfidfTransformer()),
            ('model', self.model)
        ])
        return self.pipeline

    def get_pipeline(self, ):
        '''
        Returns the sklearn pipeline for preprocessing inputs to the model
        
        Args:
            -
        Returns:
            <sklearn.pipeline.Pipeline> object
        Exception:
            -
        '''
        return self.pipeline

    def train_model(self, X_train, y_train):
        '''
        Trains the loaded model
        
        Args:
            X_train - train set as <np.array> of sentences as <str>
            y_train - labels as <np.array>
        Returns:
            -
        Exception:
            -
        '''
        pipeline = self.form_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)
        print("Train Accuracy: %0.2f"%(accuracy_score(y_pred, y_train)*100))

    def evaluate_model(self, X_test, y_test):
        '''
        Evaluates the model
        
        Args:
            X_test - test set as <np.array> of sentences as <str>
            y_test - test labels as <np.array>
        Returns:
            -
        Exception:
            -
        '''
        y_pred = self.pipeline.predict(X_test)

        print("Classification Report:\n")
        print(classification_report(y_test, y_pred))
        print()
        print("Confusion Matrix:\n")
        print (confusion_matrix(y_test, y_pred))
        print()
        print("Test Accuracy: %0.2f"%(accuracy_score(y_test, y_pred)*100))
    
    def save_model(self, model_path = "models/"):
        '''
        Saves the trained model
        
        Args:
            model_path - path to save the trained model as <str>
        Returns:
            -
        Exception:
            -
        '''
        if model_path == "models/": model_path += type(self.model).__name__
        if "." not in model_path: model_path += ".model"
        pickle.dump(self.pipeline, open(model_path, 'wb'))
    
    def load_model(self, model_path):
        '''
        Loads the trained model into memory
        
        Args:
            model_path - path to load the trained model from as <str>
        Returns:
            -
        Exception:
            -
        '''
        self.pipeline = pickle.load(open(model_path, 'rb'))
        self.model = self.pipeline['model']

    def set_model(self, model_name):
        '''
        Sets the type of model to be used
        
        Args:
            model_name - type of the model as <str>
        Returns:
            -
        Exception:
            -
        '''
        if model_name == "MultinomialNB":
            self.model = MultinomialNB()
        elif model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier()
        elif model_name == "GradientBoostingClassifier":
            self.model = GradientBoostingClassifier()
        elif model_name == "XGBoost":
            self.model = XGBClassifier()

    def isSpam(self, sentence):
        '''
        predicts if the text is spam
        
        Args:
            sentence - input sentence as <str>
        Returns:
            spam status as <bool>
        Exception:
            -
        '''
        return self.pipeline.predict(np.array([sentence])).squeeze()==1

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default = "", help="Path to the trained model file")
    parser.add_argument("--text", type=str, help="Input text to determine if it is spam or not")
    parser.add_argument("--train", type=bool, default = False, help="Whether or not to train the model")
    parser.add_argument("--data_path", type=str, default = "data/spam.csv", help="Path to the data csv file")
    parser.add_argument("--model_type", type=str, default = "MultinomialNB", help="Type of the model to be used")
    
    args = parser.parse_args()

    # Construct model path if not provided
    if args.model_path == "":
        args.model_path = os.path.join("models", args.model_type + ".model")

    # Train the model
    if args.train:
        # Load data
        dataset = Dataset(args.data_path)
        X_train, y_train, X_test, y_test = dataset.balanced_sample()

        # create spam detetor object
        sd = SpamDetector()
        # Set the model type
        sd.set_model(args.model_type)
        # Train the set model
        sd.train_model(X_train, y_train)
        # Evaluate trained model
        sd.evaluate_model(X_test, y_test)
        # Save the model for later use
        sd.save_model(args.model_path)
        # Test
        print(sd.isSpam(args.text))

    # Test the model
    else:
        # create spam detetor object
        sd = SpamDetector()
        # Set the model type
        sd.load_model(args.model_path)
        # Test
        print(sd.isSpam(args.text))