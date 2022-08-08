from codec import Codec, BinaryCodec
from collections import defaultdict, deque
import os
from typing import IO
from engine_io import FileReader

from lexers import Document


class Algorithm:
    """
    Base class for indexing algorithms
    """

    def __init__(self, posting_codec: Codec) -> None:
        self._codec: Codec = posting_codec
        self._lexicon: dict = None

    @property
    def codec(self) -> Codec:
        """
        @return
        @desc: returns the codec used by the algorithm
        """
        return self._codec

    @property
    def lexicon(self) -> dict:
        """
        @return
        @desc: returns the lexicon
        """
        return self._lexicon

    @lexicon.setter
    def lexicon(self, lexicon: dict):
        """
            @param: lexicon
            @desc: new lexicon
        """
        self._lexicon = lexicon

    def index(self, docs: list[Document]) -> str:
        """
            indexes the document in the list

            @param: docs
            @desc: documents to index

            @return: str
            @desc: the filename of the generated posting list
        """
        raise NotImplementedError

    def merge(self, posting_filenames: deque[tuple]) -> int:
        """
            merges the partial indexes

            @param: posting_filenames
            @desc: queue of tuples of filenames and read offset for the partial indexes

            @return: int
            @desc: size of the index from the merge
        """
        raise NotImplementedError


class BSBI(Algorithm):
    """
    An implementation of the Block Sort-Base Indexing algorithm
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
        out_posting_file: IO[bytes],
    ) -> int:
        """
            merges two index files into one index file

            @param: posting_file_0
            @desc: the first index file to merge

            @param: posting_file_1
            @desc: the second index file to merge

            @param: out_posting_file
            @desc: the file to write the merge

            @return: int
            @desc: size of the merge file
        """

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
        """
            merges the partial indexes

            @param: posting_filenames
            @desc: queue of tuples of filenames and read offset for the partial indexes

            @return: int
            @desc: size of the index from the merge
        """
        # count for number of merge files created
        merged: int = 0

        while len(posting_filenames) > 1:
            # assign merge file name
            out_filename: str = f"out_file_{merged}.bin"
            merged += 1

            # open merge file for writing bytes
            out_file: IO[bytes] = open(out_filename, "wb")

            # remove two partial index file name from queue
            filename_0, _ = posting_filenames.popleft()
            filename_1, _ = posting_filenames.popleft()

            # open first partial index for reading
            file_0: IO[bytes] = open(filename_0, "rb")

            # open second partial index for reading
            file_1: IO[bytes] = open(filename_1, "rb")

            # merge the partial indexes
            self.___merge(file_0, file_1, out_file)

            # add merge file name to the queue
            posting_filenames.append((out_filename, 0))

            # release resources
            file_0.close()
            os.remove(filename_0)

            file_1.close()
            os.remove(filename_1)

            out_file.close()

        # remove the last merge file from the queue
        index_filename, _ = posting_filenames.popleft()

        # add read offset for each term to the lexicon
        self.add_offset(index_filename)
        return index_filename

    def add_offset(self, filename: str):
        """
            reads the index file and adds read offset to the lexicon

            @param: filename
            @desc: the index file name
        """
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
        """
            indexes the given list of documents

            @param: docs
            @desc: list of documents to be indexed

            @return: str
            @desc: the name of the index file
        """
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

        postings = sorted(([tid, did, freq]
                          for (tid, did), freq in posting.items()))
        filename = f"posting_{self.postings}.bin"
        self.postings += 1

        self.encode_to_file(filename, postings)
        return filename

    def encode_to_file(self, filename: str, postings: list[list[int]]) -> int:
        total_bytes = len(postings) * self.codec.posting_size
        with open(filename, "wb") as fp:
            for posting in postings:
                total_bytes -= fp.write(self.codec.encode(posting))

        return total_bytes

    def decode_from_file(self, filename: str) -> list:
        with open(filename, "rb") as fp:
            while True:
                bytes_ = fp.read(self.codec.posting_size)
                if not bytes_:
                    break
                yield self.codec.decode(bytes_)

