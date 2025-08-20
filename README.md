# Disentangling lexical and grammatical information in word embeddings

### Installation
To run the experiment, please install python3 and the following packages:
`
python -m pip install -r requirements.txt
`

### Datasets

- **data-LNfr.pkl**
Contains 25,679 lexical entries extracted from LN-fr, along with their lemma, POS tags, example sentences, token positions within each sentence, and selected grammatical features.

- **stanza-LNfr.tar.gz**
Contains all LN-fr lexical entries with their example sentences annotated with [Stanza](https://stanfordnlp.github.io/stanza/available_models.html). Extract before use:
`tar -xvzf stanza-LNfr.tar.gz`

These datasets are derived from the French Lexical Network (LN-fr), a handcrafted lexical resource containing 3,127 idioms, 22,551 free lexemes, and 47,395 contextual sentences for these entries. The original LN-fr data is downloadable [here](https://www.ortolang.fr/market/lexicons/lexical-system-fr/v1?lang=en).

For our experiment, we focused only on lexemes. We combined the two datasets above for analysing and evaluation (see `main.py`).

### Source code
- **main.py**
This is the main entry point of the project. Running this script will execute the full pipeline of data processing and experiment.
- **disentangle.py**
This script contains helper functions (e.g., process) that are imported and used inside main.py. Users do not need to run this file directly, but it must be kept in the same directory so that it can be imported properly.












