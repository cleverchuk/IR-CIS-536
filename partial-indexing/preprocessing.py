from collections import OrderedDict
import string
from typing import Iterable
from nltk.stem import PorterStemmer
import re


REGEX = re.compile(r"(<\w+>|<\w+/>|\w+:[/a-zA-Z0-9-.#]*)") # regex for removing url and html tags
WORD_REGEX = re.compile(r"([`'/.,])(\w+)") # regex for removing prefixes of the first group
TOKENIZER = re.compile('(?u)\\b\\w\\w+\\b') # regex for breaking document into tokens
DOC_ID = re.compile(r"(?<=curid=)[0-9]+") # regex for extracting document id
URL = re.compile("https://en.wikipedia.org/wiki\?curid=\\d+") # regex for extracting document url

class Document:
    """
        Data structure representing Wikipedia document
    """

    def __init__(self, id, url: str, content: list[str]) -> None:
        self._content = content
        self._url = url
        self._id = id

    @property
    def content(self):
        return self._content

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._url

    def __repr__(self) -> str:
        return f"{self.id} {self.url}: {self.content[:10]}"


class Lexer:
    """
        A Lexer for the corpus
    """
    def __init__(self) -> None:
        self.stemmer: PorterStemmer = PorterStemmer()
        self.words: set = set()
        self._dictionary: OrderedDict = OrderedDict()

    def word_tokenize(self, content: str) -> list[str]:
        content_ = content.lower()
        content_ = REGEX.sub("", content_)
        # remove these [`'/.,] prefixes and stop words(words of 3 characters or less)
        content_ = [WORD_REGEX.sub(r"\2", token) for token in TOKENIZER.findall(content_) if token not in string.punctuation and len(token) > 3]

        return content_

    def stem(self, tokens) -> None:
        for idx, word in enumerate(tokens):
            if word not in string.punctuation:  # ignore punctuation during stemming
                tokens[idx] = self.stemmer.stem(word)

            else:
                tokens[idx] = word

            self.words.add(tokens[idx]) # update word set

    def lex(self, doc_text: str) -> Document:
        """
            creates a data structure for the give document text
        """
        # extract the document id
        match = URL.search(doc_text)
        url = match.group()
        id: int = int(DOC_ID.findall(url)[0])

        # extract the document content and tokenize the document
        content = doc_text[match.end():]
        content = self.word_tokenize(content)
        
        # stem the tokens
        self.stem(content)       
        return Document(id, url, content)

    @property
    def dictionary(self):
        """
            lazily compute the dictionary
        """
        if self._dictionary:
            return self._dictionary

        words = sorted(self.words)
        for code, word in enumerate(words):
            self._dictionary[word] = code

        return self._dictionary

    @dictionary.setter
    def dictionary(self, dictionary):
        self._dictionary = dictionary
        

class FileWriter:
    """
        Convenience class for writing to file
    """
    @staticmethod
    def write(content: Iterable[str], path: str):
        with open(path, 'wb') as fp:
            for line in content:
                fp.write(line.encode("utf-8"))
                fp.write(b'\n')