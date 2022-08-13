import pickle
from typing import IO, Any
from cleverchuk.lib.codec import BinaryCodec, TextCodec


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
    def read_bytes(file: IO[bytes], codec: BinaryCodec | TextCodec) -> bytes:
        """
        reads a block or a line from the given file object

        @param: file
        @desc: readable file object in the byte mode

        @param: codec
        @desc: codec implementation

        @return: bytes
        @desc: byte stream
        """
        if isinstance(codec, BinaryCodec):
            return file.read(codec.posting_size)

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