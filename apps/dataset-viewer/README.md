# ThoughtSource⚡ Dataset Viewer

*You can browse datasets on the hosted version of [ThoughtSource Dataset Viewer](http://thought.samwald.info).*

![Dataset viewer example](/resources/images/dataset-viewer.PNG)

## Setup

1. Clone repository
2. Install the CoT library `pip install -e ./libs/cot`, see [here](../../libs/cot/README.md)
3. Install streamlit `pip install streamlit`
4. Run `streamlit run ./apps/dataset-viewer/app.py`

# Load a local collection json

When you have a local json that you generated CoTs for, extracted answers, run evaluations, ... you can load it with the dataset-viewer, by setting `from_local: true` and `local_path` to the path of the json in the `config.yml`.
