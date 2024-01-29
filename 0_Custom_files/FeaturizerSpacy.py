from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import spacy
import re
import sys
import os
from pathlib import Path


class ManualFeatures(TransformerMixin, BaseEstimator):

    """
    A custom transformer that extracts manual features from text using spaCy. It integrates well into scikit-learn
    pipelines and offers a variety of text features including Part-of-Speech (POS) tags, Named Entity Recognition (NER) tags,
    and basic text descriptive statistics.

    Attributes:
        spacy_model (str): The spaCy language model to be used for tokenization and other NLP tasks.
        batch_size (int): The number of documents to process at once during spaCy's pipeline processing.
        pos_features (bool): If True, extract POS tag features.
        ner_features (bool): If True, extract NER tag features.
        text_descriptive_features (bool): If True, extract basic text descriptive features such as word count, character count, etc.

    Methods:
        get_cores() -> int:
            Determines the number of CPU cores to utilize for parallel processing, aiming to optimize performance.

        get_pos_features(cleaned_text: list) -> numpy.ndarray:
            Extracts POS tag features from the text, counting the occurrences of nouns, auxiliary verbs, main verbs, and adjectives.

        get_ner_features(cleaned_text: list) -> numpy.ndarray:
            Extracts NER tag features from the text, counting the occurrences of named entities.

        get_text_descriptive_features(cleaned_text: list) -> numpy.ndarray:
            Extracts basic text descriptive features such as total word count, character count (with and without spaces),
            average word length, digit count, number count, and sentence count.

        fit(X, y=None) -> 'ManualFeatures':
            Fits the transformer to the data. This is a dummy method for scikit-learn compatibility and does not change the state of the object.

        transform(X, y=None) -> tuple:
            Transforms the provided data using the defined feature extraction methods. It returns a tuple containing a 2D numpy array of the extracted features and a list of feature names.

    Raises:
        TypeError: If the input X is not a list or a numpy array.
        Exception: For other exceptions that may occur during the transform process.
    """


    def __init__(self, spacy_model, batch_size = 64, pos_features = True, ner_features = True, text_descriptive_features = True):

        self.spacy_model = spacy_model
        self.batch_size = batch_size
        self.pos_features = pos_features
        self.ner_features = ner_features
        self.text_descriptive_features = text_descriptive_features

    def get_cores(self):
        """
        Get the number of CPU cores to use in parallel processing.
        """
        # Get the number of CPU cores available on the system.
        num_cores = os.cpu_count()
        if num_cores < 3:
            use_cores = 1
        else:
            use_cores = num_cores // 2 + 1
        return num_cores

    def get_pos_features(self, cleaned_text):

        nlp = spacy.load(self.spacy_model)
        noun_count = []
        aux_count = []
        verb_count = []
        adj_count =[]

        # Disable the lemmatizer and NER pipelines for improved performance
        disabled_pipes = ['lemmatizer', 'ner']
        with nlp.select_pipes(disable=disabled_pipes):
            n_process = self.get_cores()
            for doc in nlp.pipe(cleaned_text, batch_size=self.batch_size, n_process=n_process):
                # Extract nouns, auxiliaries, verbs, and adjectives from the document
                nouns = [token.text for token in doc if token.pos_ in ["NOUN","PROPN"]]
                auxs =  [token.text for token in doc if token.pos_ in ["AUX"]]
                verbs =  [token.text for token in doc if token.pos_ in ["VERB"]]
                adjectives =  [token.text for token in doc if token.pos_ in ["ADJ"]]

                # Store the count of each type of word in separate lists
                noun_count.append(len(nouns))
                aux_count.append(len(auxs))
                verb_count.append(len(verbs))
                adj_count.append(len(adjectives))

        # Stack the count lists vertically to form a 2D numpy array
        return np.transpose(np.vstack((noun_count, aux_count, verb_count, adj_count)))



    def get_ner_features(self, cleaned_text):
        nlp = spacy.load(self.spacy_model)
        count_ner = []

        # Disable the tok2vec, tagger, parser, attribute ruler, and lemmatizer pipelines for improved performance
        disabled_pipes = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer']
        with nlp.select_pipes(disable=disabled_pipes):
            n_process = self.get_cores()
            for doc in nlp.pipe(cleaned_text, batch_size=self.batch_size, n_process=n_process):
                ners = [ent.label_ for ent in doc.ents]
                count_ner.append(len(ners))

        # Convert the list of NER counts to a 2D numpy array
        return np.array(count_ner).reshape(-1, 1)


    def get_text_descriptive_features(self, cleaned_text):
        list_count_words = []
        list_count_characters = []
        list_count_characters_no_space = []
        list_avg_word_length = []
        list_count_digits = []
        list_count_numbers = []
        list_count_sentences = []

        nlp = spacy.load(self.spacy_model)
        disabled_pipes = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
        with nlp.select_pipes(disable=disabled_pipes):
            if not nlp.has_pipe('sentencizer'):
                nlp.add_pipe('sentencizer')
            n_process = self.get_cores()
            for doc in nlp.pipe(cleaned_text, batch_size=self.batch_size, n_process=n_process):
                count_word = len([token for token in doc if not token.is_punct])
                count_char = len(doc.text)
                count_char_no_space = len(doc.text_with_ws.replace(' ', ''))
                avg_word_length = count_char_no_space / (count_word + 1)
                count_numbers = len([token for token in doc if token.is_digit])
                count_sentences = len(list(doc.sents))

                list_count_words.append(count_word)
                list_count_characters.append(count_char)
                list_count_characters_no_space.append(count_char_no_space)
                list_avg_word_length.append(avg_word_length)
                list_count_numbers.append(count_numbers)
                list_count_sentences.append(count_sentences)

        text_descriptive_features = np.vstack((list_count_words, list_count_characters, list_count_characters_no_space, list_avg_word_length,
                                    list_count_numbers, list_count_sentences))
        return np.transpose(text_descriptive_features)


    def fit(self, X, y=None):

        return self


    def transform(self, X, y=None):

        try:
            # Check if the input data is a list or numpy array
            if not isinstance(X, (list, np.ndarray)):
                raise TypeError(f"Expected list or numpy array, got {type(X)}")


            feature_names = []

            if self.text_descriptive_features:
                text_descriptive_features = self.get_text_descriptive_features(X)
                feature_names.extend(['count_words', 'count_characters',
                                      'count_characters_no_space', 'avg_word_length',
                                      'count_numbers', 'count_sentences'])
            else:
                text_descriptive_features = np.empty(shape=(0, 0))

            if self.pos_features:
                pos_features = self.get_pos_features(X)
                feature_names.extend(['noun_count', 'aux_count', 'verb_count', 'adj_count'])
            else:
                pos_features = np.empty(shape=(0, 0))

            if self.ner_features:
                ner_features = self.get_ner_features(X)
                feature_names.extend(['ner'])
            else:
                ner_features = np.empty(shape=(0, 0))

            # Stack the feature arrays horizontally to form a single 2D numpy array
            return np.hstack((text_descriptive_features, pos_features,  ner_features)), feature_names

        except Exception as error:
            print(f'An exception occured: {repr(error)}')