from abc import abstractmethod
import string
from nltk.stem import PorterStemmer
import re

from cleverchuk.lib.dstructures import Document


# regex for removing url and html tags
REGEX = re.compile(r"(<\w+>|<\w+/>|\w+:[/a-zA-Z0-9-.#]*)")
# regex for removing prefixes of the first group
WORD_REGEX = re.compile(r"([`'/.,])(\w+)")
TOKENIZER = re.compile('(?u)\\b\\w\\w+\\b')
DOC_ID = re.compile(r"(?<=curid=)[0-9]+")
URL = re.compile("https://en.wikipedia.org/wiki\?curid=\\d+")


class AbstractLexer:
    def __init__(self) -> None:
        self._doc_stats: dict = {}

    @abstractmethod
    def lex(self, content: str) -> Document:
        raise NotImplementedError

    @abstractmethod
    def stem(self, tokens: list[str]) -> None:
        pass

    @abstractmethod
    def word_tokenize(self, query: str) -> list[str]:
        pass

    @property
    def doc_stats(self):
        return self._doc_stats


class WikiLexer(AbstractLexer):
    """
        A Lexer for the corpus
    """

    def __init__(self) -> None:
        super().__init__()
        self.stemmer: PorterStemmer = PorterStemmer()

    def update_stats(self, id: int, content: str) -> None:
        self.doc_stats[id] = len(content)

    def word_tokenize(self, content: str) -> list[str]:
        content_ = content.lower()
        content_ = REGEX.sub("", content_)
        content_ = [WORD_REGEX.sub(r"\2", token) for token in TOKENIZER.findall(
            content_) if token not in string.punctuation and len(token) > 3]

        return content_

    def stem(self, tokens: list[str]) -> None:
        """
            removes the stem from the token in tokens
        """
        for idx, word in enumerate(tokens):
            if word not in string.punctuation:  # ignore punctuation during stemming
                tokens[idx] = self.stemmer.stem(word)

            else:
                tokens[idx] = word

    def lex(self, doc_text: str) -> Document:
        match = URL.search(doc_text)
        url = match.group()
        id: int = int(DOC_ID.findall(url)[0])

        content = doc_text[match.end():]
        self.update_stats(id, content)
        content = self.word_tokenize(content)

        self.stem(content)
        return Document(id, url, content)
