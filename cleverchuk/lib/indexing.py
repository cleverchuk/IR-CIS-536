import random
from collections import deque
import os
from statistics import mean
from time import time
from typing import IO
from cleverchuk.lib.codec import Codec, TextCodec
from cleverchuk.lib.algorithm import BSBI, Algorithm
from cleverchuk.lib.fs import FilePickler, FileReader
from cleverchuk.lib.fs import AbstractFileFactory, DefaultFileFactory

from cleverchuk.lib.lexers import AbstractLexer, WikiLexer

from sklearn.feature_extraction.text import TfidfVectorizer

# maximum number of terms to compare when we encounter unknown term
MAX_TERMS = int(os.getenv("MAX_TERMS", 10))


class Index:
    """
    The document index used for fast look up of postings for a given term
    """

    def __init__(
        self, lexicon: dict, posting_file: IO[bytes], doc_stats: dict, codec: Codec
    ):
        self.lexicon: dict = lexicon
        self.doc_stats: dict = doc_stats
        self.posting_file: IO[bytes] = posting_file

        self.codec = codec
        self._avg_dl = None
        self.vectorizer = TfidfVectorizer()

    def doc_length(self, doc_id: int):
        return self.doc_stats[doc_id]

    @property
    def avgdl(self):
        if self._avg_dl:
            return self._avg_dl

        self._avg_dl = mean(self.doc_stats.values())
        return self._avg_dl

    @property
    def corpus_size(self):
        return len(self.doc_stats)

    def release(self) -> None:
        self.posting_file.close()

    def fetch_index_record(self, term: str) -> tuple[list[list[int]], int]:
        term = term if term in self.lexicon else self.handle_unknown_term(term)
        _, doc_freq, offset = self.lexicon[term]

        self.posting_file.seek(offset)
        for _ in range(doc_freq):
            bytes_: bytes = FileReader.read_bytes(
                self.posting_file, self.codec)
            yield (self.codec.decode(bytes_), doc_freq)

    def compute_similarity(self, term0: str, term1: str) -> float:
        """
            calculates the cosine similarity of the two terms
            @param term0
            @desc: first term 

            @param term1
            @desc: second term

            @return float
            @desc: range [0, 1]
        """
        try:
            tfidf = self.vectorizer.fit_transform([term0, term1])
            return (tfidf * tfidf.T).A[0, 1]
        except:
            return 0

    def handle_unknown_term(self, term: str) -> str:
        best_match = None
        best_score = -1
        #FIXME maybe use terms that start with same character or find a better approximation?

        for term in self.lexicon:
            score = self.compute_similarity(term, term)
            if score > best_score:
                best_match = term
                best_score = score

        return best_match


class Indexer:
    """
    The indexer
    """

    def __init__(
        self, algo: Algorithm = BSBI(DefaultFileFactory(), TextCodec()), lexer: AbstractLexer = WikiLexer(), fileFactory: AbstractFileFactory = DefaultFileFactory()
    ) -> None:
        self.algo: Algorithm = algo
        self._lexer: AbstractLexer = lexer
        self._lexicon_filename = "lexicon.bin"

        self._terms_lexicon_filename = "terms_lexicon.bin"
        self._doc_stat_filename = "doc_stats.bin"
        self._index_filename = "index.bin"

        self._indexed = False
        self._index: Index = None
        self.fileFactory = fileFactory

    @property
    def index_filename(self):
        return self._index_filename

    @property
    def lexicon_filename(self):
        return self._lexicon_filename

    @property
    def terms_lexicon_filename(self):
        return self._terms_lexicon_filename

    @property
    def doc_stat_filename(self):
        return self._doc_stat_filename

    @property
    def codec(self):
        return self.algo.codec

    @property
    def lexer(self):
        return self._lexer

    @property
    def indexed(self):
        return self._indexed

    @property
    def index(self):
        return self._index

    def execute(self, filenames: list[str], block_size: int = 33554432, n: int = -1) -> Index:
        """
            indexes the corpus in represented by @param filenames

            @param: filenames
            @desc: list of files to index

            @param: block_size
            @desc: number of lines to read per system call

            @param: n
            @desc: total number of blocks to read. -1 means read all blocks
        """
        if self.indexed:
            return

        posting_filenames: deque[str] = deque()
        for filename in filenames:
            for docs_ in FileReader.read_docs(filename, block_size, n):
                block = []
                for doc in docs_:
                    block.append(self.lexer.lex(doc.strip()))

                posting_filenames.appendleft(self.algo.index(block))

        index_filename = self.algo.merge(posting_filenames)
        index_file: IO[bytes] = self.fileFactory.create(index_filename)

        # create an Index data structure for fast search
        self._indexed = True
        self._index = Index(
            self.algo.lexicon,
            index_file,
            self.lexer.doc_stats,
            self.codec,
        )

        return self._index

    def export_index(self):
        """
            exports lexicon, term lexicon and document stats using the pickle protocol
        """
        FilePickler.dump(self.index.lexicon, self._lexicon_filename)
        FilePickler.dump(self.index.doc_stats, self._doc_stat_filename)
        FilePickler.dump(self.algo.term_lexicon, self._terms_lexicon_filename)


if __name__ == "__main__":
    filenames = ["tiny_wikipedia.txt"]
    indexer: Indexer = Indexer()
    begin = time()
    indexer.index(filenames)
    end = time()
    print(f"Time using Textcodec: {end - begin}")
    os.remove("index.bin")

    indexer: Indexer = Indexer(BSBI())
    begin = time()
    indexer.index(filenames)
    end = time()
    print(f"Time using Binary codec: {end - begin}")
    os.remove("index.bin")
