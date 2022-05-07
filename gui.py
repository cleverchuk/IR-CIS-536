import PySimpleGUI as sg
from engine import Engine

SEARCH = "Search"
DOC_LIST = "DOC-LIST"
SEARCH_BOX = "SEARCH_BOX"

def format_docs(docs: list[tuple]) -> list[tuple]:
    if not docs:
        return ["No relevant document found. Please refine your query."]
        
    for i in range(len(docs)):
        id, score = docs[i]
        docs[i] =  f"URL: https://en.wikipedia.org/wiki?curid={id}, Score: {round(score, 2)}"
    
    return docs

# Initialize the search engine
engine: Engine = Engine("wikicorpus.txt")

# Add a touch of color
sg.theme("DarkAmber")

# All the stuff inside your window.
layout = [
    [sg.Text("Enter query"), sg.InputText(key=SEARCH_BOX), sg.Button(SEARCH)],
    [sg.Listbox(values=[], enable_events=False, size=(70, 10), key=DOC_LIST)],
]

# Create the Window
window = sg.Window(
    "CleverChuk's Search Engine", layout=layout, font=("arial", 12, "normal")
)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
        break
    elif event == SEARCH:
        docs = engine.search(values[SEARCH_BOX])
        window[DOC_LIST].update(format_docs(docs))

window.close()
