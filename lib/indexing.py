from lib.codec import Codec, TextCodec
from collections import defaultdict, deque
import os
from statistics import mean
from time import time
from typing import IO, Any
import pickle
from lib.algorithm import BSBI, Algorithm
from lib.engine_io import FilePickler, FileReader

from lib.lexers import AbstractLexer, WikiLexer


class Index:
    """
    The document index used for fast look up of postings for a given term
    """

    def __init__(
        self, lexicon_path: str, posting_file: IO[bytes], doc_stats_path: str, codec: Codec
    ):
        with open(lexicon_path, "rb") as fp:
            self.lexicon: dict = defaultdict(
                lambda: (-1, 0, 0), pickle.load(fp))

        with open(doc_stats_path, "rb") as fp:
            self.doc_stats: dict = pickle.load(fp)

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

        self._doc_stat_filename = "doc_stats.bin"
        self._index_filename = "index.bin"

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

    def index(self, filenames: list[str], block_size: int = 33554432, n: int = -1):
        files = os.listdir()
        if self.index_filename in files:
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
        FilePickler.dump(dict(self.algo.lexicon), self._lexicon_filename)

        FilePickler.dump(self.lexer.doc_stats, self._doc_stat_filename)


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
