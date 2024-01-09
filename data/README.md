# Games, Issues, Agents, and Rules: How to create your own structured negotiation games!
___
Our negotiation protocol was designed to make it as easy as possible to experiment with custom negotiation scenarios 
**without writing code**. The only input needed comes in the form of simple `.YAML` files as described below.

## Basics
A structured negotiation game consists of two agents playing a game according to some protocol. In our case, the agents 
are parameterized using language models (LM-agents). A game consists of a description, some issue(s), and a 
preference ordering. For example:
- a. [game setting] Alice and Bob are ordering a pizza
- b. [issues] How should they divide the slices? how much cheese should be on the pizza?
- c. [issue preferences] How important is each issue to Alice/Bob?

### games, issues, agents, and rules
The `Game`, `Issues`, `Agent` and `Rules` objects are all represented using simple `.YAML` files each detailed below. 
What is great about this setup, is that these objects can be reused and mixed for different scenarios. For example, 
an agent description for a skilled secret agent can be used both for the pizza game, or for a high-stakes hostage 
negotiation. Similarly, an issue describing 'service costs' can be used both for a rental agreement game or a loan 
agreement at a bank.

### hydra (super basics)
We use [`hydra`](https://hydra.cc/) to standardize and scale experimentation. While `hydra` is awesome, it takes some 
time to get familiar with. We therefore tried to limit user exposure to the absolute bare minimum:

```yaml
game:
  rules: data/general_game_rules.yaml
  name: <data/games/your-game-file>.yaml                         # note: place your game-file under data/games/
  issues: [<your-issue-1>.yaml, <your-issue-2>.yaml]             # note: issue files should be placed under data/issues/
  issue_weights: # preference weights for each agent             # note: all-one weights means no preference
    - [1. 1]
    - [1, 1]
  scale:  # total utility available per agent across issues      # note: we default to 100 for easy interpretation
    - 100
    - 100

agent_1:
  agent_description: data/agents/<your-agent-file-1>
  generation_parameters: ${models_dir}/openai_35_0_0.yaml    # specify model you would like to use (data/model_settings/

agent_2:
  agent_description: data/agents/<your-agent-file-2>
  generation_parameters: ${models_dir}/openai_35_0_0.yaml

negotiation_protocol:
  start_agent_index: 0  # which agent should start the negotiations
```
There are two files of interest:
- `src/configs/experiments/<your-experiment>.yaml`
- `src/configs/inference_root.yaml`

The first describes the experiment you would like to run. The second points `hydra` to your experiment by modifying the 
following line:
```yaml
defaults:
  - _self_
  - hydra: inference
  - experiments: <your-experiment>
```
Simply hitting `python src/run.py` in the command line will run your experiment and save results to `logs` folder.

## Game File
A game-file describes the setting of the negotiation game. As discussed in our work, different settings can lead to 
'inter-game' bias. That is, even though two games might have the same underlying optimization problem, one might be 
perceived as easier or more difficult than the other. Changing game-file descriptions are a promising way to test 
how robust agent performance is. An example with the four required fields are shown below:
```yaml
name: generic-loan-agreement
description: A loan officer at a bank and a prospective customer are negotiating an unsecured consumer loan agreement.
sides:
  - You are an advisor representing the best interests of the loan officer. 
  - You are an advisor representing the best interests of the prospective customer.
parties:
  - Loan Officer
  - Prospective Customer
```
In our work, we predominantly experimented with LM-agents taking the role of advisor. Setting the `sides` and `parties`
fields to the same value will have the LM-agent negotiate on their own behalf (See `data/games` for more examples).

## Issues
Most of the negotiations optimization problem is defined through `Issues`. Each issue represents a point to be 
negotiated and come in two forms:
1. Distributive: agents have **opposing** interests, i.e. their payoff vectors are ordered opposite of each others.
2. Compatible: agents have **aligned** interests, i.e., their payoff vectors are ordered in the same direction. 
An example with the mandatory fields is listed below:
```yaml
name: service fees
issue_type: distributive
descriptions:
  - You have to negotiate the amount of service fees that will be charged monthly.
  - You have to negotiate the amount of service fees that will be charged monthly.
payoffs:
  - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  - [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
payoff_labels:
  - ["$0", "$10", "$20", "$30", "$40", "$50", "$60", "$70", "$80", "$90", "$100"]
  - ["$0", "$10", "$20", "$30", "$40", "$50", "$60", "$70", "$80", "$90", "$100"]
```
To increase downstream parsing for resul evaluation, we recommend using simple `payoff_lables`. For example, 
instead of writing `['two hundred dollars', 'one hundred dollars']`, try `['$200', '$100']`. The more general you make 
the description, the more reusable your `Issue` becomes across different games (See `data/issues` for more examples).

## Agents
For a number of reasons related to potential biases hidden in LMs, the default agents are only described as 
'Representatives' of the negotiating parties (see `data/agents/anon.yaml`). It is easy to specify agent descriptions to 
simulate different scenarios. For example:
```yaml
internal_description:
  name: John Doe
  gender: male
  profession: secret agent
  age: 35
  personality_type: calculated

external_description:
  name: John Major
  gender: male
  age: early-thirties
```
Each agent has an `internal_description` that is only accessible to the agent itself. The `external_desription` on the
other hand is also accessible to the opposing agent. For each, the `name` field is the only one required (See 
`data/agents` for more examples).

## The Rules
The current rules are adapted from the famous 'Rio Copa' game [1], a negotiating protocol commonly used to simulate 
negotiations for didactic purposes at leading business schools:
```yaml
rules_prompt: "Never forget the following negotiation rules:"
rules:
  - Your total payoff is the sum of your payoffs on all issues. Higher payoffs are better than lower payoffs.
  - A valid agreement occurs only when all issues are decided. Partial agreements result in a total payoff to you of zero.
  - You are not allowed to accept any agreement that results in a payoff less than zero.
  - You are not allowed to deviate from or innovate with the payoffs listed on the payoff table. In other words, you cannot change your payoffs.
  - No side payments are allowed. For example, you cannot give the other negotiator your own money or other perks not listed in the payoff tables.
  - You may describe issues and elaborate on them as you see fit. However, you are not allowed to invent additional issues.
  - Never make an offer that is not part of the possible values in your payoff table.
```
It is easy to extend and/or change these rules for whatever best fits your purpose. The current rules-file can be found 
under `data/general_game_rules.yaml`.

[1] Robert Bontempo and Shanto Iyengar. Rio copa: A negotiation simulation. Columbia Caseworks, 2008

## Miscellaneous
Other files of interest that can be modified are:
- `message_prompts/`
- `note_prompts/`
- `offer_extraction_prompts/`
- `negotiation_defaults.yaml`

The first two folders contain prompts used to dictate how LM-agents should construct notes and messages. The current 
defaults were heavily tested. That being said, due to the fickle nature of prompt-engineering - it might very well be 
that you can improve the desired behavior by modifying these.

Similarly, the prompts used in the `offer_extraction_prompts/` folder guide how and LM should extract offers from the 
notes and messages generated by the LM-agents. These examples currently account for several error-correcting scenarios. 
A known limitation, is that the error-correction is currently 'too good', in that it obfuscates certain rule violations.
For example, a valid offer might be one of `{100, 200}`, and the agent offers `180` instead. In this case, the offer 
extraction LM will 'correct' the offer to be `200`. A better extractor would extract both the original, as well as the 
corrected offer value.

Finally, the `negotiation_defaults.yaml` file is used to set defaults in the experiment config files. This is to prevent
having to specify each field for each experiment.