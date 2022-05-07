from indexing import Indexer, Index
from scorer import Scorer


class Engine:
    """
        This is the search engine class
    """
    def __init__(self, corpus_path: str) -> None:
        self.indexer: Indexer = Indexer()
        self.corpus_path: str = corpus_path
        self.__indexed: bool = False

        self.scorer: Scorer = None

    def search(self, query: str) -> list[tuple]:
        """
            Performs lazy indexing on the first search and subsequent searches are super fast
            @param: query 
            @description: query used to find relevant documents

            @return: list[tuple]
            @description: a list of document ids and score pair           

        """
        if self.__indexed:
            # return the documents that are relevant to the query
            return self.scorer.relevant_docs(query)
        self.indexer.index(self.corpus_path) # index the corpus

        # create an Index data structure for fast search
        index: Index = Index(
            self.indexer.lexicon_filename,
            self.indexer.index_filename,
            self.indexer.doc_stat_filename,
            self.indexer.codec,
        )

        # update flag to true to avoid reindexing
        self.__indexed = True
        
        # initialize the scorer with the index and the lexer
        self.scorer = Scorer(index, self.indexer.lexer)
        return self.scorer.relevant_docs(query)
