<p align="center">
      <br/>
            <img src="assets/img/main-no-header.png" alt="image" width="800" height="auto">
      <br/>
<p>
<p align="center">
    <a href="https://epfl-dlab.github.io/LAMEN">
    <img alt="Website" src="https://img.shields.io/badge/website-online-green">
    </a>
    <a href="https://vminvsky.com">
    <img alt="Blog post" src="https://img.shields.io/badge/blog-online-green">
    </a>
    <a href="https://www.python.org/downloads/release/python-3110/"><img alt="PyPi version" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://vminvsky.com">
    <img alt="Documentation" src="https://img.shields.io/badge/arXiv-2308.01285-b31b1b.svg">
    </a>
</p>

<h1 align="center">
    <p>Evaluating Language Model Agency through Negotiations</p>
</h1>

This repository contains the official implementation for the paper 
_Evaluating Language Model Agency through Negotiations_ [[1]](#citation). 

**TL;DR: let language models negotiate to find out how well they function as autonomous agents. (🤖💬🤝🤖💬)**

Interested in learning more?
- 🎓 We wrote a paper
- 🗂️ Shared our data
- 📝 And wrote a blogpost

## Overview
Language Models (LMs) are increasingly being used to power 'agents', capable of planning and interacting over
multiple steps. Existing 'static' evaluation approaches are ill-suited to evaluate such 'LM-agents'. 
Specifically, they **do not**:
1. 🔄     Measure behavior over extended periods
2. 🤖💬🤖   Allow for cross-model interaction
3. 🤝⚖️🏆 Jointly evaluate alignment and performance metrics

Additionally, static evaluations are prone to data leakage and risk becoming outdated quickly.
Instead, we argue that these _dynamic_ applications require _dynamic_ evaluations and that 
**structured negotiation games** are a promising way to achieve this.

What can you find in this repository?
- [**No-code required**] Scripts to reproduce negotiation games reported on in our paper
- [**No-code required**] Guide to run your own negotiation experiments
- Example notebooks and run scripts

## Getting Started!
The simplest way to get started is to:
1. clone this repository, then
2. create a `secrets.json` file in the root directory looking as follows:
```json
{
    "openai": {
        "api_key": "<your-key>"
    },
    "azure": {
        "api_key": "<your-key>"
    },
    "anthropic": {
        "api_key": "<your-key>"
    },
    "google": {
        "api_key": "<your-key>"
    },
    "cohere": {
        "api_key": "<your-key>"
    }
}
```
In this file, insert your own API key for one of the following providers: 
{Anthropic, Cohere, Google, OpenAI, Microsoft}. This `secrets.json` file is part of the `.gitignore`, to prevent you 
from accidentally pushing your raw keys to GitHub :). (see 'note' below if using Google/Azure models)

Next, create a virtual environment and install the packages listed in `requirements.txt`. Once this is done you're all 
set! Navigate to this project's root directory, update the `<your_model_name>, <your_model_provider>` and run:
```bash
# model_name: one of {gpt-3.5-turbo, gpt-4, claude-2.0, claude-2.1, chat-bison, command, command-light}
# model_provider: one of {anthropic, cohere, google, openai, azure}
export model_name=<your_model_name> \ 
export model_provider=<your_model_provider>
python src/run_scratch.py \
++experiments.agent_1.model_name=$model_name \
++experiments.agent_1.model_provider=$model_provider \
++experiments.agent_2.model_name=$model_name \
++experiments.agent_2.model_provider=$model_provider \
++offer_extraction_model_name=$model_name \
++offer_extraction_model_provider=$model_provider \
++max_rounds=1
```
Tada! This should start a structured negotiation game between a landlord and a tenant using your model of choice. 
For a more elaborate example, e.g., on how to change the games/issues/issues_weights as well as the agent descriptions,
please check out the `example.ipynb` notebook.

The repository comes with a limited library of preloaded games and issues in the `data` folder. 
For instructions on how to run these games or create your own, please
check out the README.md guide contained in this repository under `data/`.

For any questions, feel free to open a ticket or reach out directly to [Tim](tim.davidson@epfl.ch) or 
[Venia](veniamin.veselovsky@epfl.ch) :).

### Note on Google/MSFT Azure
If you are using Google or MSFT Azure, you also need to update the relevant endpoints in 
`data/api_settings/apis.yaml`. At the time of release, the Google Vertex API does not support simple API keys in all 
regions. To get around this, you have to create a (1) service account, (2) set some permissions, (3) download a .json. 
Save the exported .json file in a file called `gcp_secrets.json` in the root directory of this project 
(also in `.gitignore`). 
[See the following docs for a walkthrough](https://cloud.google.com/iam/docs/service-accounts-create).

## License
MIT

## Citation
Please cite our work using one of the following if you end up using the repository - thanks!

```
[1] T.R. Davidson, V. Veselovsky, M. Josifoski, M. Peyrard, A. Bosselut, M. Kosinski, R. West (2023). 
Evaluating Language Model Agency through Negotiations.
```

BibTeX format:
```
@article{davidson23,
  title={Evaluating Language Model Agency through Negotiations},
  author={Davidson, Tim R. and 
          Veselovsky, Veniamin and
          Josifoski, Martin and
          Peyrard, Maxim and
          Bosselut, Antoine and
          Kosinski, Michal and
          West, Robert
          },
  journal={arXiv preprint},
  year={2023}
}
```
