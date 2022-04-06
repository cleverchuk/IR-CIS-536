import ast
from collections import defaultdict, deque
import os
from typing import IO, Any
import pickle

from preprocessing import Document, Lexer, FileWriter


class Codec:
    """
    Base class for different codec implementation
    """

    def __init__(self, size) -> None:
        self._size = size

    def decode(self, *arg, **kwargs) -> list:
        """
        decodes input to a list of posting
        """
        raise NotImplementedError

    def encode(self, *args, **kwargs) -> bytes:
        """
        encodes a list of positing to bytes
        """
        raise NotImplementedError

    @property
    def posting_size(self):
        """
        @return: int
        @description: configured posting size
        """
        return self._size


class TextCodec(Codec):
    """
    Codec implementation for encoding and decoding to ASCII
    """

    def __init__(self, size: int = 20) -> None:
        super().__init__(size)

    def encode(self, long_posting: list[int, str, int, list[tuple[int, int]]]) -> bytes:
        """
        encodes a list of positing to ASCII bytes

        @param: long_posting
        @desc: a very long posting list

        @return: bytes
        @desc: posting list as ASCII byte stream
        """

        posting_string = " ".join(
            map(lambda posting: f"({posting[0]},{posting[1]})", long_posting[3])
        )
        return f"{long_posting[0]} {long_posting[1]} {long_posting[2]} {posting_string}\n".encode(
            "utf-8"
        )

    def decode(
        self, bytes_: bytes
    ) -> list[int, str, int, list[tuple[int, int]]] | None:
        """
        decodes an ASCII byte stream to a posting list

        @param: bytes_
        @desc: ASCII byte stream to decode

        @return: list | None
        @desc: a posting list or None
        """

        if bytes_:
            decoded = bytes_.decode("utf-8").strip()
            index = decoded.find("(")
            singles = decoded[:index].strip().split()

            long_posting = [int(singles[0]), singles[1], int(singles[2])]
            couples = map(ast.literal_eval, decoded[index:].strip().split())
            long_posting.append(list(couples))

            return long_posting

        return None


class FileReader:
    """
    Convenience class for efficiently reading files
    """

    @staticmethod
    def read_docs(path: str, block_size=4096, n=-1) -> list[str]:
        """
        read n lines if n > -1 otherwise reads the whole file

        @param: path
        @desc: the file absolute or relative path

        @param: block_size
        @desc: the number lines to read

        @return: list[str]
        @desc: generator of list of individual line of the file
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
    def read_bytes(file: IO[bytes]) -> bytes:
        """
        reads a line from the given file object

        @param: file
        @desc: readable file object in the byte mode

        @return: bytes
        @desc: byte stream
        """

        return file.readline()


class FilePickler:
    """
    Convenience class for reading and writing objects as byte stream to file
    """

    @staticmethod
    def dump(data: Any, filename: str) -> None:
        """
        writes object to file

        @param: data
        @desc: object to write to file

        @param: filename
        @desc: name of file to write
        """
        with open(filename, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(filename: str) -> Any:
        """
        reads object from file

        @param: filename
        @desc: name of file to read

        @return: Any
        @desc: the object that was read from file
        """
        with open(filename, "rb") as fp:
            return pickle.load(fp)


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
        @return: Codec
        @desc: returns the codec used by the algorithm
        """
        return self._codec

    @property
    def lexicon(self) -> dict:
        """
        @return: dict
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

    def index(self, docs: list[Document], dictionary: dict) -> str:
        """
        indexes the document in the list

        @param: docs
        @desc: documents to index

        @return: str
        @desc: the filename of the generated posting list
        """
        raise NotImplementedError

    def merge(self, posting_filenames: deque[tuple[str,int]]) -> str:
        """
        merges the partial indexes

        @param: posting_filenames
        @desc: queue of tuples of filenames and read offset for the partial indexes

        @return: str
        @desc: file name of the merge
        """
        raise NotImplementedError


class BSBI(Algorithm):
    """
    An implementation of the Block Sort-Base Indexing algorithm
    posting_file structure: |term_code|term|doc_freq|(doc_id,term_freq)(,(doc_id,term_freq))*|
    """
    POSTING_COUNT: int = 0

    def __init__(self, posting_codec: Codec = TextCodec()) -> None:
        super().__init__(posting_codec)
        self.lexicon = defaultdict(lambda: (-1, 0))
        self.term_lexicon: dict = {}        

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

        # read an decoded a block from the posting files
        left: list = self.codec.decode(
            FileReader.read_bytes(posting_file_0, self.codec)
        )
        right: list = self.codec.decode(
            FileReader.read_bytes(posting_file_1, self.codec)
        )
        written: int = 0

        # merge the blocks based on the term code
        while left and right:

            if left[0] < right[0]:
                written += out_posting_file.write(self.codec.encode(left))
                bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                left = self.codec.decode(bytes_)

            elif left[0] == right[0]:
                left[2] = max(left[2], right[2]) # select the max document frequency since it's cumulative
                left[3] += right[3]
                written += out_posting_file.write(self.codec.encode(left))

                bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                left = self.codec.decode(bytes_)

                bytes_ = FileReader.read_bytes(posting_file_1, self.codec)
                right = self.codec.decode(bytes_)

            else:
                written += out_posting_file.write(self.codec.encode(right))
                bytes_ = FileReader.read_bytes(posting_file_1, self.codec)
                right = self.codec.decode(bytes_)

            if not left:
                while right:
                    written += out_posting_file.write(self.codec.encode(right))
                    bytes_ = FileReader.read_bytes(posting_file_1, self.codec)

                    right = self.codec.decode(bytes_)
                    written += self.codec.posting_size

            if not right:
                while left:
                    written += out_posting_file.write(self.codec.encode(left))
                    bytes_ = FileReader.read_bytes(posting_file_0, self.codec)
                    left = self.codec.decode(bytes_)

                    written += self.codec.posting_size

        return written

    def merge(self, posting_filenames: deque[tuple[str,int]]) -> str:
        """
        merges the partial indexes

        @param: posting_filenames
        @desc: queue of tuples of filenames and read offset for the partial indexes

        @return: str
        @desc: file name of the merge
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
            filename_0, pos_0 = posting_filenames.popleft()
            filename_1, pos_1 = posting_filenames.popleft()

            # open first partial index for reading
            file_0: IO[bytes] = open(filename_0, "rb")
            file_0.seek(0, os.SEEK_END)

            # compute size of first partial index file
            size_0: int = file_0.tell()
            file_0.seek(pos_0)

            # open second partial index for reading
            file_1: IO[bytes] = open(filename_1, "rb")
            file_1.seek(0, os.SEEK_END)

            # compute size of second partial index file
            size_1: int = file_1.tell()
            file_1.seek(pos_1)
            
            # merge the partial indexes
            self.___merge(file_0, file_1, out_file)

            # if partial index file wasn't consumed completely add it back to queue with it's read offset
            if file_0.tell() < size_0:
                posting_filenames.appendleft((filename_0, file_0.tell()))

            if file_1.tell() < size_1:
                posting_filenames.appendleft((filename_1, file_1.tell()))

            # add merge file name to the queue
            posting_filenames.append((out_filename, 0))

            # release resources
            file_0.close()
            file_1.close()
            out_file.close()

        # remove the last merge file from the queue
        index_filename, _ = posting_filenames.popleft()
        return index_filename

    def index(self, docs: list[Document], dictionary: dict) -> str:
        """
            Indexes the given document list

            @param: docs
            @desc: list of documents to index

            @param: dictionary
            @desc: the dictionary containing term -> code mapping

            @return: str
            @desc: the posting file name
        """
        posting = defaultdict(int) # initialize empty posting dict
        for doc in docs:
            for term in doc.content:
                term_code, doc_freq = self.lexicon[term] # lookup term code and document frequency from lexicon
                if term_code == -1:
                    term_code = dictionary[term]
                    self.term_lexicon[term_code] = term # create code -> term mapping

                posting_key = (term_code, doc.id)
                if posting_key not in posting:
                    doc_freq += 1 # compute document frequency

                posting[posting_key] += 1
                self.lexicon[term] = (term_code, doc_freq) # update lexicon with document frequency

        long_posting = defaultdict(list) # empty positing for merging
        for (term_code, doc_id), term_freq in posting.items():
            term = self.term_lexicon[term_code]
            term_code, doc_freq = self.lexicon[term]

            key = (term_code, term, doc_freq)
            long_posting[key].append((doc_id, term_freq)) # merging the posting list into one long list

        # sort list by term code
        long_posting = sorted(
            (
                [term_code, term, doc_freq, posting]
                for (term_code, term, doc_freq), posting in long_posting.items()
            ),
            key=lambda lp: lp[0],
        )

        filename = f"posting_{BSBI.POSTING_COUNT}.bin"
        BSBI.POSTING_COUNT += 1
        
        self.encode_to_file(filename, long_posting) # write to file using the provided encoding
        return filename # return posting file name

    def encode_to_file(self, filename: str, postings: list[int, str, int, list[tuple[int, int]]]) -> int:
        """
            writes the given posting list to file

            @param: filename
            @desc: name of file to write

            @param: postings
            @desc: list of long postings

            @return: int
            @desc: total number of bytes written to file
        """
        total_bytes = 0
        with open(filename, "wb") as fp:
            for posting in postings:
                total_bytes += fp.write(self.codec.encode(posting))

        return total_bytes

    def decode_from_file(self, filename: str) -> list[int, str, int, list[tuple[int, int]]]:
        """
            reads long posting list from the given file name

            @param: filename
            @desc: name of file to write

            @return: list
            @desc: list of long postings
        """        
        with open(filename, "rb") as fp:
            while True:
                bytes_ = fp.readline()
                if not bytes_:
                    break
                yield self.codec.decode(bytes_)


class Indexer:
    """
    The indexer
    """

    def __init__(self, algo: Algorithm = None, lexer: Lexer = None) -> None:
        if algo is None:
            algo = BSBI(TextCodec())

        if lexer is None:
            lexer = Lexer()

        self.algo: Algorithm = algo
        self._lexer: Lexer = lexer

    @property
    def codec(self):
        return self.algo.codec

    @property
    def lexer(self):
        return self._lexer

    def index(self, file_path, block_size=33554432, n=-1) -> None:
        """
            creates an index for the given file

            @param: file_path
            @desc: path to file to be indexed

            @param: block_size
            @desc: the size of block to read

            @param: n
            @desc: the number of blocks to read. -1 to read the all blocks
        """
        files = os.listdir()
        index_filename = f"index{os.path.splitext(file_path)[-1][1:]}.txt" # avoid index an already indexed file
        if index_filename in files:
            return

        posting_filenames: deque = deque()
        block = []
        # the indexing loop
        for docs in FileReader.read_docs(file_path, block_size, n):
            block = []
            for doc in docs:
                block.append(self.lexer.lex(doc.strip())) # process blocks into documents using lexer

            posting_filenames.appendleft(
                (self.algo.index(block, self.lexer.dictionary), 0) # index the documents using indexing algorithm
            )

        merged_filename = self.algo.merge(posting_filenames) # merge partial indexes
        os.rename(merged_filename, index_filename) # rename the full index


class Driver:
    """
    Convenience class for running the algorithm
    """
    def __init__(self, filenames: list[str]) -> None:
        self.filenames: list[str] = filenames
        self.lexer = Lexer()
        self.__make_dictionary(filenames)

    def __make_dictionary(self, filenames: list[str]) -> None:
        """
            create/loads the dictionary
        """
        dict_filename = "dictionary.txt"
        filenames_ = os.listdir()
        if dict_filename in filenames_: # try not recreate the dictionary every time
            code = 0
            dictionary: dict = dict()
            for lines in FileReader.read_docs(dict_filename, block_size=1):
                term, *_ = lines
                dictionary[term.strip()] = code
                code += 1

            self.lexer.dictionary = dictionary

        else: # create the dictionary if we haven't done so
            for filename in filenames:
                for docs_ in FileReader.read_docs(filename, 33554432):
                    for doc in docs_:
                        self.lexer.lex(doc.strip())

            FileWriter.write(self.lexer.dictionary.keys(), "dictionary.txt")

    def drive(self) -> None:
        """
            index all file in filenames
        """
        for filename in self.filenames:
            Indexer(lexer=self.lexer).index(filename) # create a new indexer every time to avoid mixing indexes since the algo and lexer are stateful


if __name__ == "__main__":
    os.chdir("partial-indexing/wikidata") # switch to directory with all the files to index
    filenames = filter(lambda name: name.startswith("wikidata"), os.listdir()) # select only the ones with wikidata in the name

    driver: Driver = Driver(filenames) # initialize driver with the filenames
    driver.drive() # drive away!!
