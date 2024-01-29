from sklearn.base import BaseEstimator, TransformerMixin
from bs4 import BeautifulSoup
import re
import spacy
import numpy as np
from nltk.stem.porter import PorterStemmer
import os

class SpacyPreprocessor(BaseEstimator, TransformerMixin):

    """
    A text preprocessor that utilizes spaCy for efficient and flexible NLP. Designed as a part of a scikit-learn
    pipeline, it provides a wide range of text cleaning and preprocessing functionalities.

    Attributes:
        model (str): The spaCy language model to be used for tokenization and other NLP tasks.
        batch_size (int): The number of documents to process at once during spaCy's pipeline processing.
        lemmatize (bool): If True, lemmatize tokens.
        lower (bool): If True, convert all characters to lowercase.
        remove_stop (bool): If True, remove stopwords.
        remove_punct (bool): If True, remove punctuation.
        remove_email (bool): If True, remove email addresses.
        remove_url (bool): If True, remove URLs.
        remove_num (bool): If True, remove numbers.
        stemming (bool): If True, apply stemming to tokens (mutually exclusive with lemmatization).
        add_user_mention_prefix (bool): If True, add '@' as a separate token (useful for user mentions in social 
            media data).
        remove_hashtag_prefix (bool): If True, do not separate '#' from the following text.
        basic_clean_only (bool): If True, perform only basic cleaning (HTML tags removal, line breaks, etc.) 
            and ignore other preprocessing steps.

    Methods:
        basic_clean(text: str) -> str:
            Performs basic cleaning of the text such as removing HTML tags and excessive whitespace.
        
        spacy_preprocessor(texts: list) -> list:
            Processes a list of texts through the spaCy pipeline with specified preprocessing options.
        
        fit(X, y=None) -> 'SpacyPreprocessor':
            Fits the preprocessor to the data. This is a dummy method for scikit-learn compatibility and does not 
            change the state of the object.
        
        transform(X, y=None) -> list:
            Transforms the provided data using the defined preprocessing pipeline. Performs basic cleaning, 
            and if `basic_clean_only` is False, it applies advanced spaCy preprocessing steps.
    
    Raises:
        ValueError: If both 'lemmatize' and 'stemming' are set to True.
        ValueError: If 'basic_clean_only' is True but other processing options are also set to True.
        TypeError: If the input X is not a list or a numpy array.
    """

    def __init__(self, model, *, batch_size = 64, lemmatize=True, lower=True, remove_stop=True,
                remove_punct=True, remove_email=True, remove_url=True, remove_num=False, stemming = False,
                add_user_mention_prefix=True, remove_hashtag_prefix=False, basic_clean_only=False):

        self.model = model
        self.batch_size = batch_size
        self.remove_stop = remove_stop
        self.remove_punct = remove_punct
        self.remove_num = remove_num
        self.remove_url = remove_url
        self.remove_email = remove_email
        self.lower = lower
        self.add_user_mention_prefix = add_user_mention_prefix
        self.remove_hashtag_prefix = remove_hashtag_prefix
        self.basic_clean_only = basic_clean_only

        if lemmatize and stemming:
            raise ValueError("Only one of 'lemmatize' and 'stemming' can be True.")

        # Validate basic_clean_only option
        if self.basic_clean_only and (lemmatize or lower or remove_stop or remove_punct or remove_num or stemming or
                                      add_user_mention_prefix or remove_hashtag_prefix):
            raise ValueError("If 'basic_clean_only' is set to True, other processing options must be set to False.")

        # Assign lemmatize and stemming

        self.lemmatize = lemmatize
        self.stemming = stemming

    def basic_clean(self, text):
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = re.sub(r'[\n\r]', ' ', text)
        return text.strip()

    def spacy_preprocessor(self, texts):
        final_result = []
        nlp = spacy.load(self.model)

        # Disable unnecessary pipelines in spaCy model
        if self.lemmatize:
            # Disable parser and named entity recognition
            disabled_pipes = ['parser', 'ner']
        else:
            # Disable tagger, parser, attribute ruler, lemmatizer and named entity recognition
            disabled_pipes = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

        with nlp.select_pipes(disable=disabled_pipes):
          # Modify tokenizer behavior based on user_mention_prefix and hashtag_prefix settings
          if self.add_user_mention_prefix or self.remove_hashtag_prefix:
              prefixes = list(nlp.Defaults.prefixes)
              if self.add_user_mention_prefix:
                  prefixes += ['@']  # Treat '@' as a separate token
              if self.remove_hashtag_prefix:
                  prefixes.remove(r'#')  # Don't separate '#' from the following text
              prefix_regex = spacy.util.compile_prefix_regex(prefixes)
              nlp.tokenizer.prefix_search = prefix_regex.search

          # Process text data in parallel using spaCy's nlp.pipe()
          for doc in nlp.pipe(texts, batch_size=self.batch_size):
              filtered_tokens = []
              for token in doc:
                  # Check if token should be removed based on specified filters
                  if self.remove_stop and token.is_stop:
                      continue
                  if self.remove_punct and token.is_punct:
                      continue
                  if self.remove_num and token.like_num:
                      continue
                  if self.remove_url and token.like_url:
                      continue
                  if self.remove_email and token.like_email:
                      continue

                  # Append the token's text, lemma, or stemmed form to the filtered_tokens list
                  if self.lemmatize:
                      filtered_tokens.append(token.lemma_)
                  elif self.stemming:
                      filtered_tokens.append(PorterStemmer().stem(token.text))
                  else:
                      filtered_tokens.append(token.text)

              # Join the tokens and apply lowercasing if specified
              text = ' '.join(filtered_tokens)
              if self.lower:
                  text = text.lower()
              final_result.append(text.strip())

        return final_result


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            if not isinstance(X, (list, np.ndarray)):
                raise TypeError(f'Expected list or numpy array, got {type(X)}')

            x_clean = [self.basic_clean(text).encode('utf-8', 'ignore').decode() for text in X]

            # Check if only basic cleaning is required
            if self.basic_clean_only:
                return x_clean  # Return the list of basic-cleaned texts

            x_clean_final = self.spacy_preprocessor(x_clean)
            return x_clean_final

        except Exception as error:
            print(f'An exception occurred: {repr(error)}')