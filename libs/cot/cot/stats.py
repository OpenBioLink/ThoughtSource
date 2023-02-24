import re
import subprocess
from collections import Counter, defaultdict
from itertools import chain

import pandas as pd
import spacy
from nltk.util import ngrams
from plotly import express as px
from plotly import graph_objects as go
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

        if key == "cot":
            lens = []
            if result[key] > 0:
                lens.append(result[key])
            for generated_cot in example["generated_cot"]:
                sentences = split_sentences(generated_cot["cot"].lower())
                toks = [tok for sent in sentences for tok in sent]
                if len(toks) > 0:
                    lens.append(len(toks))
            if len(lens) > 0:
                result[key] = sum(lens) / len(lens)

    return result


def get_n_grams_counter(example, counter, key, N):
    if key == "cot":
        text = " ".join(example[key])
    else:
        text = example[key]
    if text is not None:
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
                            if key == "generated_cot":
                                counters["na"][name]["generated_cot"] += len(val)
                                counters["na"][name]["question_with_generated_cot"] += 1
                            else:
                                counters["na"][name][key] += 1
                        if key == "type":
                            counters["types"][val] += 1
                            counters["types_datasets"][val].add(name)
                    progress.update(task2, advance=1.0)
                progress.update(task1, advance=1.0)
            progress.update(task0, advance=1.0)
        progress.refresh()
    return counters


def _generate_ngrams_data(collection, key, N):
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
                    get_n_grams_counter(entry, n_gram_counter, key, N)
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

    data = [
        (
            name,
            data_dict["train"].num_rows if "train" in data_dict else "-",
            data_dict["validation"].num_rows if "validation" in data_dict else "-",
            data_dict["test"].num_rows if "test" in data_dict else "-",
        )
        for name, data_dict in collection
    ]
    table_num_examples = pd.DataFrame.from_records(data, columns=["Name", "Train", "Valid", "Test"])
    _print_table(table_num_examples)

    data = []
    for key, counter in counters["na"].items():
        data.append(
            [key] + [counter[ckey] for ckey in ["question", "choices", "cot", "generated_cot", "question_with_generated_cot", "answer"]]
        )
    table_nan = pd.DataFrame.from_records(
        data, columns=["dataset", "question", "choices", "cot", "generated_cot", "question_with_generated_cot", "answer"]
    )
    _print_table(table_nan)

    data = []
    for key, count in counters["types"].items():
        data.append([key, count, counters["types_datasets"][key]])
    table_types = pd.DataFrame.from_records(data, columns=["type", "number samples", "datasets"])
    _print_table(table_types)

    return (table_num_examples, table_nan, table_types)


def prepare_overlap_matrix(collection, key, N):
    n_gram_counters = _generate_ngrams_data(collection, key, N)
    n_grams_merge = {}
    for name, n_grams in n_gram_counters.items():
        n_grams_merge[name] = set([item for counters in n_grams.values() for item in counters.keys()])

    data = []
    datasets = sorted(n_grams_merge.keys())

    for idx_x, name_x in enumerate(datasets):
        vals = []
        for idx_y, name_y in enumerate(datasets):
            if idx_x == idx_y:
                jacc = 1.0
            elif idx_x > idx_y:
                if len(n_grams_merge[name_x]) > 0 and len(n_grams_merge[name_y]) > 0:
                    inters = len(n_grams_merge[name_x].intersection(n_grams_merge[name_y]))
                    uni = len(n_grams_merge[name_x].union(n_grams_merge[name_y]))
                    jacc = inters / min(len(n_grams_merge[name_x]), len(n_grams_merge[name_y]))
                else:
                    jacc = 0.0
            else:
                jacc = None
            vals.append(jacc)
        data.append(vals)
    return n_grams_merge, data


def plot_dataset_overlap(collection, N=3):
    """
    It takes the n-grams from each dataset and calculates the Jaccard similarity between each pair of
    datasets

    :param data: the data dictionary returned by the function generate_data
    """

    from plotly.subplots import make_subplots

    subpl = make_subplots(rows=1, cols=2, subplot_titles=("Question", "CoT"), print_grid=False)

    for index, key in enumerate(["question", "cot"]):
        n_grams_merge, data = prepare_overlap_matrix(collection, key, N)

        # because of inversed y axis
        data.reverse()

        fig = go.Heatmap(
            z=data,
            x=list(sorted(n_grams_merge.keys())),
            y=list(sorted(n_grams_merge.keys(), reverse=True)),
            hoverongaps=False,
            coloraxis="coloraxis",
            text=[["" if (element is None or element < 0.01) else f"{element:.2f}" for element in row] for row in data],
            texttemplate="%{text}",
        )
        subpl.add_trace(fig, row=1, col=index + 1)

    subpl.update_layout(height=700, width=1400)
    subpl.update_layout(coloraxis=dict(colorscale="tempo", cmin=0.0, cmax=1.0), showlegend=False)
    subpl.for_each_xaxis(lambda x: x.update(showgrid=False, zeroline=False))
    subpl.for_each_yaxis(lambda x: x.update(showgrid=False, zeroline=False))
    subpl.write_image(f"dataset_overlap.svg")
    subpl.write_image(f"dataset_overlap.png")
    subpl.show()


def plot_token_length_distribution(collection, splits=False):
    token_len = _generate_token_length_data(collection)

    table = token_len[["dataset", "context", "question", "cot"]].groupby("dataset").agg(["max", "mean"])
    # table.columns = table.columns.map('_'.join).reset_index()
    _print_table(table)

    for key in ["context", "question", "cot"]:
        token_len_ = token_len[token_len[key] > 0]
        fig = px.box(
            token_len_,
            x=key,
            y="dataset",
            color="split" if splits else None,
            labels={
                "dataset": "Dataset",
                "cot": "Number of tokens in CoT",
                "question": "Number of tokens in question",
                "context": "Number of tokens in context",
            },
            width=1100,
            points=False,
        )
        fig.write_image(f"token_length_distribution_{key}.svg")
        fig.write_image(f"token_length_distribution_{key}.png")
        fig.show()
    return (table, fig)


def get_n_outlier(dataset, field="cot", n=5):
    outlier = []
    for example in dataset:
        if field == "cot":
            txt = " ".join(example["cot"]).lower()
        else:
            txt = example[field].lower()

        sents = split_sentences(txt)
        toks = [tok for sent in sents for tok in sent]
        # outlier.append((toks, len(toks), example["id"]))
        outlier.append((example, len(toks)))
    outlier = sorted(outlier, key=lambda x: x[1], reverse=True)
    return (outlier[:n], outlier[-n:])

def evaluation_as_table(eval:dict):
    import pandas as pd
    eval_dict = pd.json_normalize(eval).to_dict('records')[0]
    eval_list = list(eval_dict.keys())
    datasets = sorted(list(eval.keys()))

    models = []
    prompts = []
    for i in eval_list:
        dataset,split,metric,model,prompt = i.split(".")
        if model not in models:
            models.append(model)
        if prompt not in prompts:
            prompts.append(prompt)
            
    models = sorted(models)

    if "None_None_None" in prompts: prompts.remove("None_None_None")

    # no instructions implemented yet
    # instructions = []
    cot_triggers = []
    for i in prompts:
        instruction, cot_trigger, _ = i.split("_")
        # if instruction not in instructions:
        #     instructions.append(instruction)
        if cot_trigger not in cot_triggers:
            cot_triggers.append(cot_trigger)

    cot_triggers = sorted(cot_triggers)

    cot_trigger_header = sorted(cot_triggers*len(models))
    model_header = models*len(cot_triggers)
    header = [cot_trigger_header, model_header]
    df = pd.DataFrame(columns=header, index=datasets)

    for k,v in eval_dict.items():
        dataset,split,metric,model,prompt = k.split(".")
        instruction, cot_trigger, _ = prompt.split("_")
        df.loc[dataset, (cot_trigger, model)] = round(v,2)

    df.dropna(how='all', axis=1, inplace=True)

    return df
