from lib.codec import Codec, TextCodec
from collections import deque
import os
from statistics import mean
from time import time
from typing import IO
from lib.algorithm import BSBI, Algorithm
from lib.engine_io import FilePickler, FileReader

from lib.lexers import AbstractLexer, WikiLexer


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

    def fetch_docs(self, term: str) -> tuple[list[list[int]], int]:
        _, doc_freq, offset = self.lexicon[term]

        self.posting_file.seek(offset)
        for _ in range(doc_freq):
            bytes_: bytes = FileReader.read_bytes(
                self.posting_file, self.codec)
            yield (self.codec.decode(bytes_), doc_freq)


class Indexer:
    """
    The indexer
    """

    def __init__(
        self, algo: Algorithm = BSBI(TextCodec()), lexer: AbstractLexer = WikiLexer()
    ) -> None:
        self.algo: Algorithm = algo
        self._lexer: AbstractLexer = lexer
        self._lexicon_filename = "lexicon.bin"

        self._terms_lexicon_filename = "terms_lexicon.bin"
        self._doc_stat_filename = "doc_stats.bin"
        self._index_filename = "index.bin"

        self._indexed = False
        self._index: Index = None

    @property
    def index_filename(self):
        return self._index_filename

    @property
    def lexicon_filename(self):
        return self._lexicon_filename

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
        if self.indexed:
            return

        posting_filenames: deque = deque()
        for filename in filenames:
            for docs_ in FileReader.read_docs(filename, block_size, n):
                block = []
                for doc in docs_:
                    block.append(self.lexer.lex(doc.strip()))

                posting_filenames.appendleft((self.algo.index(block), 0))
        
        index_filename = self.algo.merge(posting_filenames)
        os.rename(index_filename, self.index_filename)
        index_file: IO[bytes] = open(self.index_filename, "rb")

        # create an Index data structure for fast search
        self._indexed = True
        self._index = Index(
            dict(self.algo.lexicon),
            index_file,
            self.lexer.doc_stats,
            self.codec,
        )

        return self._index 

    def export_index(self): 
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
