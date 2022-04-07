# Partial Indexing of Wikipedia Corpus
The code in this branch is the checkpoint-2 in the process of creating a search engine for the Wikipedia Corpus.

# How-To Run
- install python 3.10+
- execute: pip install -r requirements.txt
- change working directory to the directory containing the code
- change *line 560* in indexing.py to the fully qualified path of the directory containing the wikidata files to be indexed. ensure that the file names has *wikidata* as substring
- execute indexing.py using: python3 indexing.py