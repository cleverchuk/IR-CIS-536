from collections import defaultdict, deque
import os
from signal import raise_signal
from typing import IO
from black import out

from numpy import block, byte

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
    def __init__(self, size = 12) -> None:
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
                posting.append(int.from_bytes(bytes_[i : i + _bytes_], byteorder="little"))

            return posting

        return None


class TextCodec(Codec):
    def __init__(self, size = 20) -> None:
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
    def read_docs(path: str, block_size=4096, n=10):
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
        if  isinstance(codec, BinaryCodec):
            return file.read(codec.posting_size)
        
        return file.readline()


class Algorithm:
    def __init__(self, posting_codec: Codec) -> None:
        self._codec = posting_codec

    @property
    def codec(self):
        return self._codec

    def index(self, docs: list[Document]) -> str:
        raise NotImplementedError

    def merge(
        self,
        posting_file_0: IO[bytes],
        posting_file_1: IO[bytes],
        out_posting_file: IO[bytes],
        out_size: int = 409600,
    ) -> int:
        raise NotImplementedError


class BSBI(Algorithm):
    """
    posting_file structure: |term_id(4)|doc_id(4)|term_freq(4)|
    """

    def __init__(self, posting_codec: Codec = BinaryCodec()) -> None:
        super().__init__(posting_codec)
        self.lexicon: dict = defaultdict(lambda: (-1, 0, 0))
        self.terms: int = 0

        self.postings = 0

    def merge(
        self,
        posting_file_0: IO[bytes],
        posting_file_1: IO[bytes],
        out_posting_file: IO[bytes],
        out_size: int,
    ) -> int:

        left: list = self.codec.decode(FileReader.read_bytes(posting_file_0, self.codec))
        right: list = self.codec.decode(FileReader.read_bytes(posting_file_1, self.codec))
        written: int = 0

        while left and right and written < out_size:
            written += self.codec.posting_size

            if left[0] < right[0] or (left[0] == right[0] and left[1] < right[1]):
                out_posting_file.write(self.codec.encode(left))
                bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                left = self.codec.decode(bytes_)

            else:
                out_posting_file.write(self.codec.encode(right))
                bytes_ = FileReader.read_bytes(posting_file_1, self.codec)
                right = self.codec.decode(bytes_)

            if not left and written < out_size:
                while right and written < out_size:
                    out_posting_file.write(self.codec.encode(right))
                    bytes_ = FileReader.read_bytes(posting_file_1, self.codec)

                    right = self.codec.decode(bytes_)
                    written += self.codec.posting_size
            
            if not right and written < out_size:
                while left and written < out_size:
                    out_posting_file.write(self.codec.encode(left))
                    bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                    left = self.codec.decode(bytes_)
                    
                    written += self.codec.posting_size
                          


        return written

    def index(self, docs: list[Document]) -> str:
        posting = defaultdict(int)
        for doc in docs:
            for term in doc.content:
                term_id, doc_freq, global_term_freq = self.lexicon[term]
                if term_id == -1:
                    term_id = self.terms
                    self.terms += 1

                posting_key = (term_id, doc.id)
                if posting_key not in posting:
                    doc_freq += 1

                posting[posting_key] += 1
                self.lexicon[term] = (term_id, doc_freq, global_term_freq + 1)

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


class Driver:
    """
    A driver class for putting it all together
    """

    def run(self, file_path):
        lexer = Lexer()
        bsbi = BSBI(TextCodec())

        posting_filenames: deque = deque()
        block = []
        for docs_ in FileReader.read_docs(file_path):
            block = []
            for doc in docs_:
                block.append(lexer.lex(doc.strip()))

            posting_filenames.appendleft((bsbi.index(block), 0))

        self.merge(bsbi, posting_filenames)

    def merge(self, algo: Algorithm, posting_filenames: deque[tuple]):
        merged = 0
        out_size = 1048576

        while len(posting_filenames) > 1:
            out_filename = f"out_file_{merged}.bin"
            merged += 1

            out_file = open(out_filename, "wb")

            filename_0, pos_0 = posting_filenames.popleft()
            filename_1, pos_1 = posting_filenames.popleft()

            file_0 = open(filename_0, "rb")
            file_0.seek(0, os.SEEK_END)

            size_0 = file_0.tell()
            file_0.seek(pos_0)

            file_1 = open(filename_1, "rb")
            file_1.seek(0, os.SEEK_END)

            size_1 = file_1.tell()
            file_1.seek(pos_1)
            algo.merge(file_0, file_1, out_file, out_size)

            if file_0.tell() < size_0:
                posting_filenames.appendleft((filename_0, file_0.tell()))

            if file_1.tell() < size_1:
                posting_filenames.appendleft((filename_1, file_1.tell()))

            posting_filenames.append((out_filename, 0))
            
            file_0.close()
            file_1.close()
            out_file.close()


if __name__ == "__main__":
    file_path = "inverted-index/tiny_wikipedia.txt"
    Driver().run(file_path)
