from collections import OrderedDict, defaultdict
import pprint
import string
from typing import Counter, List
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

REGEX = re.compile("(<\w+>)|<\w+/>")

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
        self._dictionary: OrderedDict = OrderedDict()
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

        wiki_doc = WikiDoc(url, REGEX.sub("", content))        
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


class FileWriter:
    def write(self, content: List[str], path: str):
        with open(path, 'w') as fp:
            fp.writelines(content)


class Driver:
    def run(self, file_path):
        fr = FileReader()
        fw = FileWriter()
        parser = Parser()

        processors = []
        for line in fr.read_lines(file_path):
            wiki_doc = parser.parse(line)
            processors.append(WikiDocProcessor(wiki_doc))
        
        corpus_processor = WikiCorpusProcessor(processors)
        corpus_processor.process(PorterStemmer())
        fw.write(corpus_processor.dictionary.keys(), "dictionary.txt")

        unigram  = []
        for word, code in corpus_processor.dictionary.items():
            line = f"{code} {word} {corpus_processor.term_freq[word]} {corpus_processor.doc_freq[word]}"
            unigram.append(line)

        fw.write(unigram, "unigrams.txt")


if __name__ == "__main__":
    file_path = "tiny_wikipedia.txt"
    Driver().run(file_path)