from cmath import log
from collections import Counter, defaultdict
from cleverchuk.lib.lexers import *
from cleverchuk.lib.indexing import *


class Scorer:
    """
        An implementation of BM25 scoring formula
    """
    def __init__(self, index: Index, lexer=WikiLexer()) -> None:
        self.lexer = lexer
        self.index = index

    def relevant_docs(self, query: str, k: int = 10) -> list[tuple]:
        """
        tokenizes the query and retrieve all relevant document

        @param: query
        @desc: the user query

        @param: k
        @desc: the number documents to return default to ten

        @return: list
        @desc: list of document id and score pair
        """
        tokens = self.lexer.word_tokenize(query)
        self.lexer.stem(tokens)
        q_freq = Counter(tokens)

        scores = defaultdict(int)
        for term in tokens:
            for posting, doc_freq in self.index.fetch_index_record(term):
                _, did, freq = posting
                scores[did] += self.score(
                    q_freq[term],
                    freq,
                    self.index.avgdl,
                    self.index.doc_length(did),
                    doc_freq,
                    self.index.corpus_size,
                )

        first_k = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]
        return first_k

    def score(
        self,
        query_freq: int,
        term_freq: int,
        avgdl: int,
        doc_length: int,
        doc_freq: int,
        corpus_size: int,
    ) -> float:
        """
        compute the BM25 score using the given parameters

        @param: query_freq
        @desc: the number of times the term appeared in the query

        @param: term_freq
        @desc: the number of times the term appeared in the document

        @param: avgdl
        @desc: the average document length

        @param: doc_length
        @desc: the length of the document

        @param: doc_freq
        @desc: the number of documents that the term appeared in

        @param: corpus_size
        @desc: the number of documents in the corpus

        @return: float
        @desc: the score of the document for the term being considered
        """
        k = 5 # abitrary value. you can pick any that works for your dataset
        b = 0.5 # abitrary value. you can pick any that works for your dataset

        idf = log((corpus_size + 1) / doc_freq).real
        numerator = query_freq * (k + 1) * term_freq * idf
        denom = term_freq + k * (1 - b + b * doc_length / avgdl)

        return numerator / denom
