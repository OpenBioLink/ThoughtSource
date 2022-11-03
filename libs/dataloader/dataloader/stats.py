import re
import subprocess
from collections import Counter, defaultdict
from itertools import chain

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from nltk.util import ngrams
from rich.progress import Progress

# download language package if not already installed
if not spacy.util.is_package("en_core_web_sm"):
    _ = subprocess.run(
        "spacy download en_core_web_sm --quiet",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
STOPWORDS = nlp.Defaults.stop_words
re_sent_ends_naive = re.compile(r"[.\n]")
re_stripper_naive = re.compile(r"[^a-zA-Z\.\n]")


def splitter_naive(x):
    return re_sent_ends_naive.split(re_stripper_naive.sub(" ", x))


# list of tokens for one sentence
def remove_stop_words(text):
    result = []
    for w in text:
        if w not in STOPWORDS:
            result.append(w)
    return result


def split_sentences(txt):
    """Naive sentence splitter that uses periods or newlines to denote sentences."""
    sentences = (x.split() for x in splitter_naive(txt) if x)
    sentences = list(map(remove_stop_words, list(sentences)))
    return sentences


def get_n_grams(sentences, N):
    ng = (ngrams(x, N) for x in sentences if len(x) >= N)
    return list(chain(*ng))


def get_tuples_manual_sentences(txt, N):
    if not txt:
        return [], []


def get_token_length_per_examples(example):
    result = {}
    for key in ["context", "question", "cot"]:
        if key == "cot":
            text = " ".join(example[key])
        else:
            text = example[key]
        if text is None:
            result[key] = 0
        else:
            sentences = split_sentences(text.lower())
            toks = [tok for sent in sentences for tok in sent]
            result[key] = len(toks)
    return result


def get_n_grams_counter(example, counter, N):
    result = {}
    for key in ["context", "question", "cot"]:
        if key == "cot":
            text = " ".join(example[key])
        else:
            text = example[key]
        if text is None:
            result[key] = 0
        else:
            sentences = split_sentences(text.lower())
            ngrams = get_n_grams(sentences, N)
            tups = ["_".join(tup) for tup in ngrams]
            counter.update(tups)


def isna(val):
    if val is None:
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    if isinstance(val, str) and val == "":
        return True
    return False


def _generate_counter_data(collection):
    counters = dict()

    with Progress() as progress:
        task0 = progress.add_task("[blue]Dataset...", total=len(collection))
        task1 = progress.add_task("[red]Split...")
        task2 = progress.add_task("[green]Example...")

        counters["types"] = Counter()
        counters["types_datasets"] = defaultdict(set)
        counters["na"] = dict()

        for name, dataset in collection:
            progress.reset(task1, total=len(dataset))
            counters["na"][name] = Counter()
            for split, data in dataset.items():
                progress.reset(task2, total=len(data))
                for entry in data.flatten():
                    # count na's and types
                    # iterates only over first lvl
                    for key, val in entry.items():
                        if not isna(val):
                            counters["na"][name][key] += 1
                        if key == "type":
                            counters["types"][val] += 1
                            counters["types_datasets"][val].add(name)
                    progress.update(task2, advance=1.0)
                progress.update(task1, advance=1.0)
            progress.update(task0, advance=1.0)
        progress.refresh()
    return counters


def _generate_ngrams_data(collection, N):
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
                    get_n_grams_counter(entry, n_gram_counter, N)
                    progress.update(task2, advance=1.0)
                n_grams_counters[name][split] = n_gram_counter
                progress.update(task1, advance=1.0)
            progress.update(task0, advance=1.0)
        progress.refresh()
    return n_grams_counters


def _generate_token_length_data(collection):
    import pandas as pd

    hist_data = []

    with Progress() as progress:
        task0 = progress.add_task("[blue]Dataset...", total=len(collection))
        task1 = progress.add_task("[red]Split...")
        task2 = progress.add_task("[green]Example...")

        for name, dataset in collection:
            progress.reset(task1, total=len(dataset))
            for split, data in dataset.items():
                progress.reset(task2, total=len(data))
                for entry in data:
                    result = get_token_length_per_examples(entry)
                    result["total_token_length"] = sum([v for k, v in result.items()])
                    result["split"] = split
                    result["dataset"] = name
                    hist_data.append(result)
                    progress.update(task2, advance=1.0)
                progress.update(task1, advance=1.0)
            progress.update(task0, advance=1.0)

        progress.refresh()

        return pd.DataFrame(hist_data)


def _print_table(table):
    try:
        display  # noqa
    except NameError:
        print(table)
    else:
        display(table)  # noqa


def display_stats_tables(collection):
    counters = _generate_counter_data(collection)
    data = []
    for key, counter in counters["na"].items():
        data.append(
            [key] + [counter[ckey] for ckey in ["question", "choices", "cot", "answer"]]
        )
    table = pd.DataFrame.from_records(
        data, columns=["dataset", "question", "choices", "cot", "answer"]
    )
    _print_table(table)

    data = []
    for key, count in counters["types"].items():
        data.append([key, count, counters["types_datasets"][key]])
    table = pd.DataFrame.from_records(
        data, columns=["type", "number samples", "datasets"]
    )
    _print_table(table)


def plot_dataset_overlap(collection, N=3):
    """
    It takes the n-grams from each dataset and calculates the Jaccard similarity between each pair of
    datasets

    :param data: the data dictionary returned by the function generate_data
    """
    n_gram_counters = _generate_ngrams_data(collection, N)
    n_grams_merge = {}
    for name, n_grams in n_gram_counters.items():
        n_grams_merge[name] = set(
            [item for counters in n_grams.values() for item in counters.keys()]
        )

    n_grams_merge

    data = []
    for name_x in sorted(n_grams_merge.keys(), reverse=True):
        vals = []
        for name_y in sorted(n_grams_merge.keys()):
            if name_x != name_y:
                inters = len(n_grams_merge[name_x].intersection(n_grams_merge[name_y]))
                uni = len(n_grams_merge[name_x].union(n_grams_merge[name_y]))
                jacc = inters / uni
            else:
                jacc = None
            vals.append(jacc)
        data.append(vals)

    fig = go.Figure(
        data=go.Heatmap(
            z=data,
            x=list(sorted(n_grams_merge.keys())),
            y=list(sorted(n_grams_merge.keys(), reverse=True)),
            hoverongaps=False,
        )
    )
    fig.update_layout(
        autosize=False,
        width=700,
        height=700,
    )
    fig.show()


def plot_token_length_distribution(collection):
    token_len = _generate_token_length_data(collection)
    for key in ["context", "question", "cot"]:
        fig = px.box(token_len, x=key, y="dataset", color="split")
        fig.show()
