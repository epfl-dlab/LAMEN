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
- fire (used for running scripts)
- aiohttp (not yet, used for async api calls)

NOTE: older versions of Python could work but are not tested

## Structure
- data/
  - agent_descriptions/: .json agent files, e.g., age, backstory, profession, etc.
  - games/: .json files of games (see `game_creator.py`)
  - results/: outputs of running experiments
- src
  - `agents.py`: agent and negotiation classes
  - `run_file.py`: run experiments
  - `utils.py`: helper functions
  - `game_creator.py`: (heavy TODOs) create games in structured format
  - `dlabchain/`: various helper files to query LLM REST APIs
- `explorations.ipynb`: notebook to scratch on
- `secrets.json`: local file to host API keys

## Quick Start
TODO

## Questions:
Please ping Venia & Tim on slack