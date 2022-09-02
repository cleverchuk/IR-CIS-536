from abc import abstractmethod
import os
from typing import IO, AnyStr, Iterable, Any
import pickle
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


class AbstractFile(IO[bytes]):
    def __init__(self, file_path: str, mode: str = "rb") -> None:
        ...

    @abstractmethod
    def remove(self) -> None:
        raise NotImplementedError

    def read(self, __n: int = ...) -> AnyStr:
        raise NotImplementedError

    def readline(self, __limit: int = ...) -> AnyStr:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def write(self, __s: AnyStr) -> int:
        raise NotImplementedError

    def writelines(self, __lines: Iterable[AnyStr]) -> None:
        raise NotImplementedError

    def seek(self, __offset: int, __whence: int = ...) -> int:
        raise NotImplementedError


class AbstractFileFactory:
    @abstractmethod
    def create(self, file_path, mode="rb") -> AbstractFile:
        raise NotImplementedError


class LocalFile(AbstractFile):
    def __init__(self, file_path: str, mode: str = "rb") -> None:
        self.file = open(file_path, mode=mode)

    def read(self, __n: int = ...) -> AnyStr:
        return self.file.read(__n)

    def readable(self) -> bool:
        return self.file.readable()

    def readline(self, __limit: int = ...) -> AnyStr:
        if __limit == ...:
            return self.file.readline()
        return self.file.readline(__limit)

    def readlines(self, __hint: int = ...) -> list[AnyStr]:
        return self.file.readlines(__hint)

    def close(self) -> None:
        self.file.close()

    def remove(self) -> None:
        self.close()
        os.remove(self.file.name)

    def writable(self) -> bool:
        return self.file.writable()

    def write(self, __s: AnyStr) -> int:
        return self.file.write(__s)

    def writelines(self, __lines: Iterable[AnyStr]) -> None:
        return self.file.writelines(__lines)

    def seek(self, __offset: int, __whence: int = ...) -> int:
        return self.file.seek(__offset)


class DefaultFileFactory(AbstractFileFactory):
    def create(self, file_path, mode="rb") -> AbstractFile:
        return LocalFile(file_path, mode)
