from abc import abstractmethod
from io import FileIO
import os
from typing import IO, AnyStr, Iterable


class AbstractFile(IO[bytes]):
    def __init__(self, file_path: str, mode: str = "rb") -> None:
        ...

    @abstractmethod
    def remove(self) -> None:
        ...

class AbstractFileFactory:
    @abstractmethod
    def create(self, file_path, mode="rb") -> AbstractFile:
        ...


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