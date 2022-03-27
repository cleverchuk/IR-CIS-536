from collections import defaultdict, deque
import os
from re import L
from statistics import mean
from typing import IO, Any
import pickle

from preprocessing import Document, Lexer


class Codec:
    def __init__(self, size) -> None:
        self._size = size

    def decode(self, *arg, **kwargs) -> list:
        raise NotImplementedError

    def encode(self, *args, **kwargs) -> bytes:
        raise NotImplementedError

    @property
    def posting_size(self):
        return self._size


class BinaryCodec(Codec):
    def __init__(self, size=12) -> None:
        super().__init__(size)

    def encode(self, posting: list[int]) -> bytes:
        return (
            posting[0].to_bytes(4, byteorder="little")
            + posting[1].to_bytes(4, byteorder="little")
            + posting[2].to_bytes(4, byteorder="little")
        )

    def decode(self, bytes_: bytes, posting_size=12, _bytes_=4) -> list:
        if bytes_:
            posting = []
            for i in range(0, posting_size, _bytes_):
                posting.append(
                    int.from_bytes(bytes_[i : i + _bytes_], byteorder="little")
                )

            return posting

        return None


class TextCodec(Codec):
    def __init__(self, size=20) -> None:
        super().__init__(size)

    def encode(self, posting: list[int]) -> bytes:
        return f"{posting[0]} {posting[1]} {posting[2]}\n".encode("utf-8")

    def decode(self, bytes_: bytes) -> list:
        if bytes_:
            decoded = bytes_.decode("utf-8").strip()
            return list(map(int, decoded.split()))

        return None


class FileReader:
    """
    Convenience class for efficiently reading files
    """

    @staticmethod
    def read_docs(path: str, block_size=4096, n=-1):
        """
        read n lines if n > -1 otherwise reads the whole file
        """
        with open(path) as fp:
            while True:
                lines = fp.readlines(block_size)
                if lines and n:
                    yield lines
                else:
                    break
                n -= 1

    @staticmethod
    def read_bytes(file: IO[bytes], codec: BinaryCodec | TextCodec) -> bytes:
        if isinstance(codec, BinaryCodec):
            return file.read(codec.posting_size)

        return file.readline()


class FilePickler:
    @staticmethod
    def dump(data: Any, filename: str) -> None:
        with open(filename, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(filename: str) -> Any:
        with open(filename, "rb") as fp:
            return pickle.load(fp)


class Algorithm:
    def __init__(self, posting_codec: Codec) -> None:
        self._codec: Codec = posting_codec
        self._lexicon: dict = None

    @property
    def codec(self):
        return self._codec

    @property
    def lexicon(self):
        return self._lexicon

    @lexicon.setter
    def lexicon(self, value):
        self._lexicon = value

    def index(self, docs: list[Document]) -> str:
        raise NotImplementedError

    def merge(self, posting_filenames: deque[tuple]) -> int:
        raise NotImplementedError


class BSBI(Algorithm):
    """
    posting_file structure(bin): |term_id(4)|doc_id(4)|term_freq(4)|
    """

    def __init__(self, posting_codec: Codec = BinaryCodec()) -> None:
        super().__init__(posting_codec)
        self.lexicon = defaultdict(lambda: (-1, 0))
        self.term_lexicon: dict = {}

        self.terms: int = 0
        self.postings: int = 0

    def ___merge(
        self,
        posting_file_0: IO[bytes],
        posting_file_1: IO[bytes],
        out_posting_file: IO[bytes]
    ) -> int:

        left: list = self.codec.decode(
            FileReader.read_bytes(posting_file_0, self.codec)
        )
        right: list = self.codec.decode(
            FileReader.read_bytes(posting_file_1, self.codec)
        )
        written: int = 0

        while left and right:
            written += self.codec.posting_size

            if left[0] < right[0] or (left[0] == right[0] and left[1] < right[1]):
                out_posting_file.write(self.codec.encode(left))
                bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                left = self.codec.decode(bytes_)

            else:
                out_posting_file.write(self.codec.encode(right))
                bytes_ = FileReader.read_bytes(posting_file_1, self.codec)
                right = self.codec.decode(bytes_)

            if not left:
                while right:
                    out_posting_file.write(self.codec.encode(right))
                    bytes_ = FileReader.read_bytes(posting_file_1, self.codec)

                    right = self.codec.decode(bytes_)
                    written += self.codec.posting_size

            if not right:
                while left:
                    out_posting_file.write(self.codec.encode(left))
                    bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                    left = self.codec.decode(bytes_)

                    written += self.codec.posting_size

        return written

    def merge(self, posting_filenames: deque[tuple]) -> str:
        merged: int = 0

        while len(posting_filenames) > 1:
            out_filename: str = f"out_file_{merged}.bin"
            merged += 1

            out_file: IO[bytes] = open(out_filename, "wb")

            filename_0, pos_0 = posting_filenames.popleft()
            filename_1, pos_1 = posting_filenames.popleft()

            file_0: IO[bytes] = open(filename_0, "rb")
            file_0.seek(0, os.SEEK_END)

            size_0: int = file_0.tell()
            file_0.seek(pos_0)

            file_1: IO[bytes] = open(filename_1, "rb")
            file_1.seek(0, os.SEEK_END)

            size_1: int = file_1.tell()
            file_1.seek(pos_1)
            self.___merge(file_0, file_1, out_file)

            if file_0.tell() < size_0:
                posting_filenames.appendleft((filename_0, file_0.tell()))

            if file_1.tell() < size_1:
                posting_filenames.appendleft((filename_1, file_1.tell()))

            posting_filenames.append((out_filename, 0))

            file_0.close()
            file_1.close()
            out_file.close()

        index_filename, _ = posting_filenames.popleft()
        self.add_offset(index_filename)
        return index_filename

    def add_offset(self, filename: str):
        with open(filename, "rb") as fp:
            offset = 0
            prev_id = -1
            while True:
                bytes_: bytes = FileReader.read_bytes(fp, self.codec)
                if not bytes_:
                    break

                term_id, *_ = self.codec.decode(bytes_)
                if prev_id != term_id:
                    term_id, doc_freq = self.lexicon[self.term_lexicon[term_id]]
                    self.lexicon[self.term_lexicon[term_id]] = (
                        term_id,
                        doc_freq,
                        offset,
                    )

                prev_id = term_id
                offset += len(bytes_)

    def index(self, docs: list[Document]) -> str:
        posting = defaultdict(int)
        for doc in docs:
            for term in doc.content:
                term_id, doc_freq = self.lexicon[term]
                if term_id == -1:
                    term_id = self.terms
                    self.term_lexicon[term_id] = term
                    self.terms += 1

                posting_key = (term_id, doc.id)
                if posting_key not in posting:
                    doc_freq += 1

                posting[posting_key] += 1
                self.lexicon[term] = (term_id, doc_freq)

        postings = sorted([[tid, did, freq] for (tid, did), freq in posting.items()])
        filename = f"posting_{self.postings}.bin"
        self.postings += 1

        self.encode_to_file(filename, postings)
        return filename

    def encode_to_file(self, filename: str, postings: list[list[int]]) -> int:
        total_bytes = len(postings) * 12
        with open(filename, "wb") as fp:
            for posting in postings:
                total_bytes -= fp.write(self.codec.encode(posting))

        return total_bytes

    def decode_from_file(self, filename: str, posting_size: int = 12) -> list:
        with open(filename, "rb") as fp:
            while True:
                bytes_ = fp.read(posting_size)
                if not bytes_:
                    break
                yield self.codec.decode(bytes_)


class Index:
    def __init__(
        self, lexicon_path: str, posting_path: str, doc_stats_path: str, codec: Codec
    ):
        with open(lexicon_path, "rb") as fp:
            self.lexicon: dict = defaultdict(lambda: (-1, 0, 0), pickle.load(fp))

        with open(doc_stats_path, "rb") as fp:
            self.doc_stats: dict = pickle.load(fp)

        self.posting_file: IO[bytes] = open(posting_path, "rb")
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

    # FIX: use generators
    def fetch_docs(self, term: str) -> tuple[list[list[int]], int]:
        _, doc_freq, offset = self.lexicon[term]
        postings = []

        self.posting_file.seek(offset)
        for _ in range(doc_freq):
            bytes_: bytes = FileReader.read_bytes(self.posting_file, self.codec)
            postings.append(self.codec.decode(bytes_))

        return (postings, doc_freq)


class Indexer:
    """
    The indexer
    """

    def __init__(
        self, algo: Algorithm = BSBI(TextCodec()), lexer: Lexer = Lexer()
    ) -> None:
        self.algo: Algorithm = algo
        self._lexer: Lexer = lexer
        self._lexicon_filename = "lexicon.bin"
        self._doc_stat_filename = "doc_stats.bin"

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

    def index(self, file_path, block_size=33554432, n = -1):
        posting_filenames: deque = deque()
        block = []
        for docs_ in FileReader.read_docs(file_path, block_size, n):
            block = []
            for doc in docs_:
                block.append(self.lexer.lex(doc.strip()))

            posting_filenames.appendleft((self.algo.index(block), 0))

        self._index_filename = self.algo.merge(posting_filenames)
        FilePickler.dump(dict(self.algo.lexicon), self._lexicon_filename)
        FilePickler.dump(self.lexer.doc_stats, self._doc_stat_filename)
