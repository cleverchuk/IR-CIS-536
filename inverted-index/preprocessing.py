import string
from nltk.stem import PorterStemmer
import re


REGEX = re.compile(r"(<\w+>|<\w+/>|\w+:[/a-zA-Z0-9-.#]*)") # regex for removing url and html tags
WORD_REGEX = re.compile(r"([`'/.,])(\w+)") # regex for removing prefixes of the first group
TOKENIZER = re.compile('(?u)\\b\\w\\w+\\b')
DOC_ID = re.compile(r"(?<=curid=)[0-9]+")
URL = re.compile("https://en.wikipedia.org/wiki\?curid=\\d+")

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
        self.stemmer = PorterStemmer()

    def word_tokenize(self, string):
        string = string.lower()
        return TOKENIZER.findall(string)

    def lex(self, doc_text: str) -> Document:
        match = URL.search(doc_text)
        url = match.group()
        content = doc_text[match.end():]

        content = REGEX.sub("", content)
        content = [WORD_REGEX.sub(r"\2", token) for token in  self.word_tokenize(content)]
        
        for idx, word in enumerate(content):
            if word not in string.punctuation:  # ignore punctuation during stemming
                content[idx] = self.stemmer.stem(word)

            else:
                content[idx] = word

        id = int(DOC_ID.findall(url)[0])
        wiki_doc = Document(id, url, content)
        return wiki_doc