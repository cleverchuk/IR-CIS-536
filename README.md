# CleverChuk Search Engine
A simple and extensible search engine backend built while taking CIS-536 at UMICH. This library has a main.py file that you can execute following the HOW-TO below.

# How To Run
- get dataset from [here](https://drive.google.com/file/d/1oYBWxEwSvaBOB6zIhwRKBxuVoJElLi3B/view?usp=sharing)
- ensure you have python 3.10+ installed.
- execute `pip install -r requirements.txt` to get the required dependencies.
- execute `python indexing.py` to create the inverted index. This will take ~15 minutes so go do other stuff.
    - nltk might ask you to download some stuff for the stemmer so just follow the instructions.
- execute `python main.py` to interact with the gui and enter queries.
