from io import BytesIO
import io
import math


from lib.codec import BinaryCodec, CloudStorageCodec, TextCodec
import pickle
from typing import IO, Any
from google.cloud import storage


class FileIO:
    """
        Interface for writing and read file
    """

    def read(self, path: str) -> BytesIO:
        raise NotImplementedError

    def write(self, bytes: BytesIO, path: str) -> str:
        raise NotImplementedError


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


class GCloudFileIO(FileIO):
    """
        FileIO that writes and reads from Google cloud storage
    """

    def __init__(self, bucket_name) -> None:
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def read(self, path: str) -> BytesIO:
        """Downloads a blob into memory."""
        # Construct a client side representation of a blob.
        blob = self.bucket.blob(path)
        contents: str = blob.download_as_string()
        return BytesIO(contents)

    def write(self, bytes: BytesIO, path: str) -> str:
        """Uploads a file to the bucket."""
        blob = self.bucket.blob(path)
        blob.upload_from_file(bytes)

        blob.make_public()
        return blob._get_download_url(self.storage_client)


class CloudIndexExporter:
    """
        A file of this size stores 356015 postings of size 1508 bytes
    """
    CHUNK_SIZE = 536870620

    def __init__(self, fileIO: FileIO) -> None:
        self.fileIO = fileIO

    def export(self, index_filename, chunks: int = ...):
        if chunks == ...:
            with open(index_filename, "rb") as fp:
                fp.seek(io.SEEK_END)
                size = fp.tell()
                fp.seek(0)

                chunks = math.ceil(size / CloudIndexExporter.CHUNK_SIZE)
                for i in range(chunks):
                    bytes_ = fp.read(CloudIndexExporter.CHUNK_SIZE)
                    self.fileIO.write(BytesIO(bytes_), str(i))

        else:
            with open(index_filename, "rb") as fp:
                fp.seek(io.SEEK_END)
                size = fp.tell()
                fp.seek(0)

                chunk_size = math.floor(size / chunks)
                for i in range(chunks):
                    bytes_ = fp.read(chunk_size)
                    self.fileIO.write(BytesIO(bytes_), str(i))


class CloudPostingFile(IO[bytes]):
    def __init__(self, filename: str, fileIO: FileIO) -> None:
        self.fileIO = fileIO
        self.filename = filename
        self.posting_file: bytes

        self.index = 0
        self.read_index = 0

    def ensure_safety(self, offset: int) -> bool:
        pos: int = self.read_index + offset
        if self.posting_file and len(self.posting_file) > pos:
            return True

        try:
            posting_file = self.fileIO.read(str(self.index))
            self.posting_file += posting_file.getvalue()
            self.index += 1

            return True

        except Exception as e:
            print(f"Error reading posting from cloud: {e}")
            return False

    def seek(self, offset):
        if self.ensure_safety(offset):
            self.read_index += offset
        else:
            raise IndexError

    def read(self, __n: int = ...) -> bytes:
        size = CloudStorageCodec.SIZE if __n == ... else __n
        end = self.read_index + size

        if self.ensure_safety(__n):
            bytes_ = self.posting_file[self.read_index:end]
            self.read_index = end
            return bytes_

        else:
            raise EOFError

    def readline(self, __limit: int = ...) -> bytes:
        return self.read(__limit)

    def close(self):
        pass

    