class Document:
    """
        Data structure representing a document in a corpus.
        Originally reperesented a nicely structured wikipedia document.
        The only fields that're required are id and content.
    """

    def __init__(self, id, url: str, content: list[str]) -> None:
        self._content = content
        self._url = url
        self._id = id

    @property
    def content(self):
        return self._content

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._url

    def __repr__(self) -> str:
        return f"{self.id} {self.url}: {self.content[:10]}"