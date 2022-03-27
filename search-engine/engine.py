from time import time
from indexing import Indexer, Index, BSBI
from scorer import Scorer


class Engine:
    def __init__(self, corpus_path: str) -> None:
        self.indexer: Indexer = Indexer()
        self.corpus_path: str = corpus_path
        self.__indexed: bool = False

        self.scorer: Scorer = None

    def search(self, query: str) -> list[tuple]:
        if self.__indexed:
            return self.scorer.rank(query)
        self.indexer.index(self.corpus_path, block_size=4096)

        index: Index = Index(
            self.indexer.lexicon_filename,
            self.indexer.index_filename,
            self.indexer.doc_stat_filename,
            self.indexer.codec,
        )

        self.__indexed = True
        self.scorer = Scorer(index, self.indexer.lexer)
        return self.scorer.rank(query)


if __name__ == "__main__":
    engine: Engine = Engine("search-engine/tiny_wikipedia.txt")
    start = time()
    docs = engine.search("The twenty-sixth season of \"The Bachelor\" premiered on January 3, 2022")
    
    elapsed = time() - start
    print(f"{docs}: {elapsed}")
