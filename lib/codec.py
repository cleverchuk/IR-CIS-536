
from abc import abstractmethod


class Codec:
    """
    Base class for different codec implementation
    """

    def __init__(self, size) -> None:
        self._size = size

    @abstractmethod
    def decode(self, bytes_: bytes) -> list:
        """
        decodes input to a list of posting
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, posting: list[int]) -> bytes:
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


class BinaryCodec(Codec):
    """
    Codec implementation for encoding and decoding to bytes
    """

    def __init__(self, size=12) -> None:
        super().__init__(size)

    def encode(self, posting: list[int]) -> bytes:
        """
        encodes a list of positing to bytes

        @param: posting
        @desc: a posting list

        @return: bytes
        @desc: positing list as byte stream
        """
        return (
            posting[0].to_bytes(4, byteorder="big")
            + posting[1].to_bytes(4, byteorder="big")
            + posting[2].to_bytes(4, byteorder="big")
        )

    def decode(self, bytes_: bytes, posting_size=12, _bytes_=4) -> list | None:
        """
        decodes byte stream to a posting list

        @param: bytes_
        @desc: byte stream to decode

        @param: posting_size
        @desc: the size of the posting list in bytes

        @param: _bytes_
        @desc: the size of each element in the posting list in bytes

        @return: list
        @desc: a posting list or None
        """
        if bytes_:
            posting = []
            for i in range(0, posting_size, _bytes_):
                posting.append(
                    int.from_bytes(bytes_[i: i + _bytes_], byteorder="big")
                )

            return posting

        return None


class TextCodec(Codec):
    """
    Codec implementation for encoding and decoding to ASCII
    """

    def __init__(self, size=20) -> None:
        super().__init__(size)

    def encode(self, posting: list[int]) -> bytes:
        """
        encodes a list of positing to ASCII bytes

        @param: posting
        @desc: a posting list

        @return: bytes
        @desc: positing list as ASCII byte stream
        """
        return f"{posting[0]} {posting[1]} {posting[2]}\n".encode("utf-8")

    def decode(self, bytes_: bytes) -> list | None:
        """
        decodes an ASCII byte stream to a posting list

        @param: bytes_
        @desc: ASCII byte stream to decode

        @return: list | None
        @desc: a posting list or None
        """
        if bytes_:
            decoded = bytes_.decode("utf-8").strip()
            return list(map(int, decoded.split()))

        return None
