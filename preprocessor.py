import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
import re
from bs4 import BeautifulSoup


class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stemmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=100000)(EnglishStemmer().stem)
        self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize

    def __call__(self, text):
        text = re.sub(r'[\.\?\!\,\:\;\"]', ' ', text)
        # text = re.sub('[-]', ' ', text)
        text = re.sub(r'<.?p>', '', text)
        # text = BeautifulSoup(text, "lxml").text
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token.lower() for token in tokens if
                  token.isalpha()]  # removing punctuations from tokens and converting to lower case
        stop_words = stopwords.words('english')
        tokens = [token for token in tokens if not token in stop_words]
        tokens = [self.stem(token) for token in tokens]
        return tokens
