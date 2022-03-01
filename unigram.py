from collections import defaultdict
import pprint
import string
from typing import Counter, List
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class WikiDoc:
    def __init__(self, url, content) -> None:
        self._content = content
        self._url = url

    @property
    def content(self):
        return self._content

    @property
    def url(self):
        return self._url

    def __repr__(self) -> str:
        return f"{self.url}: {self.content[:10]}"


class WikiDocProcessor:
    def __init__(self, wiki_doc: WikiDoc) -> None:
        self._wiki_doc: WikiDoc = wiki_doc
        self._term_freq: dict = {}
        self._tokens: list = []

        self.__words = word_tokenize(self.wiki_doc.content)

    @property
    def term_freq(self) -> dict:
        return self._term_freq

    @property
    def wiki_doc(self) -> WikiDoc:
        return self._wiki_doc

    @property
    def tokens(self) -> list:
        return self._tokens

    def process(self, stemmer: PorterStemmer) -> None:
        for word in self.__words:
            if word not in string.punctuation:
                stem = stemmer.stem(word)
                self._tokens.append(stem)

            else:
                self._tokens.append(word)

        self._term_freq = Counter(self._tokens)


class WikiCorpusProcessor:
    def __init__(self, wiki_doc_processors: List[WikiDocProcessor]) -> None:
        self._dictionary: dict = {}
        self._term_freq: dict = {}

        self._doc_freq: defaultdict = defaultdict(int)
        self._tokens: list = []

        self._processors: List[WikiDocProcessor] = wiki_doc_processors

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def term_freq(self):
        return self._term_freq

    @property
    def doc_freq(self):
        return self._doc_freq        

    def process(self, stemmer: PorterStemmer) -> None:

        for processor in self._processors:
            processor.process(stemmer)
            self._tokens += processor.tokens
        
        self._term_freq = Counter(self._tokens)
        for token in self._term_freq.keys():
            for processor in self._processors:
                if token in processor.term_freq:
                    self._doc_freq[token] += 1

        for code, token in enumerate(self._term_freq.keys()):
            self._dictionary[token] = code


class Parser:
    def parse(self, doc_text: str) -> WikiDoc:
        tokens = doc_text.split(' ')
        url = tokens[0]
        content = " ".join(tokens[1:])

        wiki_doc = WikiDoc(url, content)        
        return wiki_doc


class FileReader:
    def read_lines(self, path: str):
        with open(path) as fp:
            while True:
                line = fp.readline().strip()
                if line:
                    yield line
                else:
                    break


class Driver:
    def run(self, file_path):
        fr = FileReader()
        parser = Parser()
        processors = []

        for line in fr.read_lines(file_path):
            wiki_doc = parser.parse(line)
            processors.append(WikiDocProcessor(wiki_doc))
        
        corpus_processor = WikiCorpusProcessor(processors)
        corpus_processor.process(PorterStemmer())

        pprint.pprint(corpus_processor.dictionary)
        pprint.pprint(corpus_processor.term_freq)
        pprint.pprint(corpus_processor.doc_freq)


if __name__ == "__main__":
    file_path = "tiny_wikipedia.txt"
    Driver().run(file_path)