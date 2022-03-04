from collections import OrderedDict, defaultdict
import pprint
import string
from typing import Counter, List, Union
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

REGEX = re.compile(r"(<\w+>|<\w+/>|\w+:[/a-zA-Z0-9-.#]*)")


class WikiDoc:
    """
        Data structure representing Wikipedia document
    """
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
    """
        A class that process a single WikiDoc computing it's term frequency
    """
    def __init__(self, wiki_doc: WikiDoc) -> None:
        self._wiki_doc: WikiDoc = wiki_doc
        self._term_freq: dict = {}
        self._tokens: list = []

        self.__words = self.wiki_doc.content

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
        """
            stems the token in WikiDoc and compute term frequency
        """
        for word in self.__words:
            if word not in string.punctuation: # ignore punctuation during stemming
                stem = stemmer.stem(word)
                self._tokens.append(stem)

            else:
                self._tokens.append(word)

        self._term_freq = Counter(self._tokens)


class WikiCorpusProcessor:
    """
        A class that computes the dictionary, global term freqency and document frequency for the corpus
    """
    def __init__(self, wiki_doc_processors: List[WikiDocProcessor]) -> None:
        self._tokens: list = []
        self._term_freq: dict = {}
        self._dictionary: OrderedDict = OrderedDict()

        self._doc_freq: defaultdict = defaultdict(int)
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
        """
            computes the dictionary, global term freqency and document frequency for the corpus
        """
        for processor in self._processors: # invoke each WikiDocProcessor to process their doc
            processor.process(stemmer)
            self._tokens += processor.tokens # update corpus tokens

        self._term_freq = Counter(self._tokens) # compute global term frequency
        for token in self._term_freq.keys():
            for processor in self._processors:
                if token in processor.term_freq:
                    self._doc_freq[token] += 1 # compute doc freqency

        words = sorted(self._term_freq.keys())
        for code, token in enumerate(words):
            self._dictionary[token] = code # compute the dictionary


class Lexer:
    """
        A Lexer for the corpus
    """
    def lex(self, doc_text: str) -> WikiDoc:
        tokens = doc_text.split(' ')
        url = tokens[0]
        content = " ".join(tokens[1:])

        content = REGEX.sub("", content)
        content = word_tokenize(content)
        wiki_doc = WikiDoc(url, content)

        return wiki_doc


class FileReader:
    """
        Convenience class for efficiently reading files
    """
    def read_lines(self, path: str, n=-1):
        """
            read n lines if n > -1 otherwise reads the whole file
        """
        with open(path) as fp:
            while True:
                line = fp.readline().strip()
                if line and n != 0:
                    yield line
                else:
                    break
                n -= 1


class FileWriter:
    """
        Convenience class for writing to file
    """
    def write(self, content: List[str], path: str):
        with open(path, 'wb') as fp:
            for line in content:
                fp.write(line.encode("utf-8"))
                fp.write(b'\n')


class StemmerLemma:
    """
        A wrapper for either a Stemmer or a Lemmatizer
    """
    def __init__(self, target: Union[PorterStemmer, WordNetLemmatizer] = PorterStemmer()) -> None:
        self.target = target

    def stem(self, word):
        if isinstance(self.target, PorterStemmer):
            return self.target.stem(word)
        elif isinstance(self.target, WordNetLemmatizer):
            return self.target.lemmatize(word)
        else:
            self.target.stem(word)

class Stemmer:
    """
        A class that applies stemming after lemmatizing
    """
    def __init__(self, lemma: WordNetLemmatizer = WordNetLemmatizer()) -> None:
        self.lemma = lemma
        self.stemmer = PorterStemmer()

    def stem(self, word):
        return self.stemmer.stem(self.lemma.lemmatize(word))


class Lemma:
    """
        A class that applies lemmatizing after stemming
    """
    def __init__(self, stemmer: PorterStemmer = PorterStemmer()) -> None:
        self.stemmer = stemmer
        self.lemma = WordNetLemmatizer()

    def stem(self, word):
        return self.lemma.lemmatize(self.stemmer.stem(word))


class Driver:
    """
        A driver class for putting it all together
    """
    def run(self, file_path):
        fr = FileReader()
        fw = FileWriter()
        lexer = Lexer()

        processors = []
        for line in fr.read_lines(file_path, 1500):
            wiki_doc = lexer.lex(line)
            processors.append(WikiDocProcessor(wiki_doc))

        corpus_processor = WikiCorpusProcessor(processors)
        corpus_processor.process(PorterStemmer())
        fw.write(corpus_processor.dictionary.keys(), "dictionary.txt")

        unigram = []
        for word, code in corpus_processor.dictionary.items():
            line = (
                code, word, corpus_processor.doc_freq[word], corpus_processor.term_freq[word])
            unigram.append(line)

        unigram = sorted(unigram, key=lambda tup: tup[3], reverse=True)
        fw.write(
            map(lambda tup: f"{tup[0]} {tup[1]} {tup[2]} {tup[3]}", unigram), "unigrams.txt")


if __name__ == "__main__":
    file_path = "tiny_wikipedia.txt"
    Driver().run(file_path)
