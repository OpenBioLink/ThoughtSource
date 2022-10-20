import spacy
import re
from nltk.util import ngrams
from itertools import chain
from rich import print as rprint
from rich.progress import Progress
import pandas as pd
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
STOPWORDS = nlp.Defaults.stop_words
re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_naive = re.compile('[^a-zA-Z\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))

# list of tokens for one sentence
def remove_stop_words(text):
    result = []
    for w in text:
        if w not in STOPWORDS:
            result.append(w)
    return result


def get_tuples_manual_sentences(txt, N):
    """Naive get tuples that uses periods or newlines to denote sentences."""
    if not txt:
        return [], []
    sentences = (x.split() for x in splitter_naive(txt) if x)
    sentences = list(map(remove_stop_words, list(sentences)))
    ng = (ngrams(x, N) for x in sentences if len(x) >= N)
    return sentences, list(chain(*ng))

def token_length_per_entry(example, counter, N):
    result = {}
    for key in ["context", "question", "cot"]:
        if key == "cot":
            text = " ".join(example[key])
        else:
            text = example[key]
        if text is None:
            result[key] = 0
        else:
            sents, ngrams = get_tuples_manual_sentences(text.lower(), N)
            toks = [tok for sent in sents for tok in sent]
            tups = ["_".join(tup) for tup in ngrams]
            counter.update(tups)
            result[key] = len(toks)
    return result, counter 

def generate_data(collection, N=3):
    import pandas as pd

    hist_data = []
    n_grams_counters = defaultdict(dict)

    with Progress() as progress:
        task0 = progress.add_task("[blue]Dataset...", total=len(collection))
        task1 = progress.add_task("[red]Split...")
        task2 = progress.add_task("[green]Example...")


        for name, dataset in collection:
            progress.reset(task1, total=len(dataset))

            for split, data in dataset.items():
                progress.reset(task2, total=len(data))
                n_gram_counter = Counter()
                for entry in data:
                    result, n_gram_counter = token_length_per_entry(entry, n_gram_counter, N)
                    result["total_token_length"] = sum([v for k, v in result.items()])
                    result["split"] = split
                    result["dataset"] = name
                    hist_data.append(result)
                    progress.update(task2, advance=1.0)
                n_grams_counters[name][split] = n_gram_counter
                progress.update(task1, advance=1.0)
            progress.update(task0, advance=1.0)


        progress.refresh()

        return {"token_len": pd.DataFrame(hist_data), "n_gram_counters": n_grams_counters}



def plot_dataset_overlap(data):
    """
    It takes the n-grams from each dataset and calculates the Jaccard similarity between each pair of
    datasets
    
    :param data: the data dictionary returned by the function generate_data
    """
    n_grams_merge = {}
    for name, n_grams in data["n_gram_counters"].items():
        n_grams_merge[name] = set([item for counters in n_grams.values() for item in counters.keys()])

    n_grams_merge

    data = []
    for name_x in sorted(n_grams_merge.keys(), reverse=True):
        vals = []
        for name_y in sorted(n_grams_merge.keys()):
            if name_x != name_y:
                inters = len(n_grams_merge[name_x].intersection(n_grams_merge[name_y]))
                uni = len(n_grams_merge[name_x].union(n_grams_merge[name_y]))
                jacc = inters/uni
            else:
                jacc = None
            vals.append(jacc)
        data.append(vals)

    fig = go.Figure(data=go.Heatmap(
                    z=data,
                    x=list(sorted(n_grams_merge.keys())),
                    y=list(sorted(n_grams_merge.keys(), reverse=True)),
                    hoverongaps = False))
    fig.update_layout(
        autosize=False,
        width=700,
        height=700,
    )
    fig.show()

def plot_token_length_distribution(data):
    for key in ["context", "question", "cot"]:
        fig = px.box(data["token_len"], x=key, y="dataset", color="split")
        fig.show()
