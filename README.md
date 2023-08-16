# GPTeam: Structured Negotiating with Large Language Models
---
## Overview
- repo to experiment with LLM-context-agents performing negotiations
- study:
  - message input/output structure
  - message/note causal model
  - agent descriptions
  - stop conditions

## Dependencies
- python >= 3.10
- requests (used to make REST API requests)
- tiktoken (used to estimate token conversion length)
- fire (maybe one day used for running scripts)
- aiohttp (not yet, used for async api calls)
- openai 
- hydra
- omegaconf 

NOTE: older versions of Python could work but are not tested

## Structure
Most of the data and configs are structured in yaml files and interact with (hydra)[https://hydra.cc/docs/intro/].

- data/
  - agent_descriptions/: .json agent files, e.g., age, backstory, profession, etc.
  - games/: .json files of games (see `game_utils.py`)
  - message_prompts/: (to-do)
  - note_prompts/: (to-do)
  - results/: outputs of running experiments
- src
  - utils/ (to-refactor) 
    - `utils.py`: helper functions
    - `game_utils.py`: (heavy TODOs) create games in structured format
    - `dlabchain/`: various helper files to query LLM REST APIs
  - `agents.py`: agent and negotiation classes
  - `run_file.py`: run experiments
  - `evaluation.py`: evaluation class
  - `protocols.py`: agent interaction protocols
- `explorations.ipynb`: notebook to scratch on
- `secrets.json`: local file to host API keys
  - `OPENAI_API_KEY`: for models from OPENAI
  - `AZURE_API_KEY`: for azure based models
- (discuss which form we prefer)`.env``: save API keys. currently used for two models


## Quick Start
TODO

## Questions:
Please ping Venia & Tim on slack