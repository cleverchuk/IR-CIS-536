from cmath import log
from collections import Counter
from preprocessing import *
from indexing import *


class Scorer:
    def __init__(self, index: Index, lexer=Lexer()) -> None:
        self.lexer = lexer
        self.index = index

    def relevant_docs(self, query: str, k: int = 10) -> list[tuple]:
        tokens = self.lexer.word_tokenize(query)
        self.lexer.stem(tokens)
        q_freq = Counter(tokens)

        scores = defaultdict(int)
        for term in tokens:
            for posting, doc_freq in self.index.fetch_docs(term):
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
        k = 5
        b = 0.5

        idf = log((corpus_size + 1) / doc_freq).real
        numerator = query_freq * (k + 1) * term_freq * idf
        denom = term_freq + k * (1 - b + b * doc_length / avgdl)

        return numerator / denom
