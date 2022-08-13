from cleverchuk.lib.indexing import Indexer, Index
from cleverchuk.engine.scorer import Scorer


class Engine:
    """
        This is the search engine class
    """

    def __init__(self, corpus_path: str, indexer: Indexer = Indexer(), index: Index = None) -> None:
        self.indexer: Indexer = indexer
        self.corpus_path: str = corpus_path
        self.scorer: Scorer = Scorer(index, indexer.lexer) if index else None

        self.index = index

    def search(self, query: str) -> list[tuple]:
        """
            Performs lazy indexing on the first search and subsequent searches are super fast
            @param: query 
            @description: query used to find relevant documents

            @return: list[tuple]
            @description: a list of document ids and score pair           

        """
        if self.indexer.indexed or self.index:
            # return the documents that are relevant to the query
            return self.scorer.relevant_docs(query)

        self.index = self.indexer.execute([self.corpus_path])  # index the corpus
        self.indexer.export_index()
        # initialize the scorer with the index and the lexer
        self.scorer = Scorer(self.indexer.index, self.indexer.lexer)
        return self.scorer.relevant_docs(query)
