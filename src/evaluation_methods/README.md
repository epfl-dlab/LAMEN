# Evaluations: Debiasing and Metrics
___
We provide a brief overview of the metrics available and the steps taken to debias results.

## Debias Games
We attempt to control for two types of bias:
1. intra-game bias
2. agent bias

As explained in our paper (section 3.1 and Appendix A.2), an unfair advantage might occur depending on:
1. The role/side represented in a negotiation, e.g., in a landlord/tenant negotiation, one of those side might be 'easier' to play for some reason.
2. The starting position during the negotiation (also known as the 'anchoring effect').

To control for these, a LM-agent plays both sides and both starting positions, after which results are averaged. This 
means that for 'self-play' negotiations, we need a minimum of two runs to obtain an unbiased output. For cross-play 
negotiations, we require four runs per unbiased output.

To minimize agent specific bias, we reduce the default agent to be a 'Representative' of one of the negotiation sides, 
not mentioning potentially biasing attributes such as gender, societal standing, or age. When designing your own agents,
 keep in mind that changing these agent attributes might bias your results. A limited study of the effects of explicitly
 mentioning an agent's negotiation experience ({average, expert, awful}) is presented in appendix C.1 of the paper.

## Metrics
We distinguish between three types of metrics (see section 2 of the paper):
1. Performance, as measured by obtained payoff utility during games
2. Faithfulness, or 'action-consistency' between stated beliefs and realized actions
3. Instruction-following, the ability to follow various instructions

### Performance
Generally, structured negotiations allow for pure conflict games and collaborative games. In the former, we have classic
 zero-sum outcomes, i.e., one utility point for agent A is minus one utility point for agent B. The latter has room for 
collaboration. In our setup, each negotiation side can gain a maximum of U=1 (normalized) utility points. For zero-sum 
games, this means equilibrium is reached when both sides have 0.5. For collaborative games, each side can obtain U>0.5 
if they manage to collaborate. 

### Faithfulness
We distinguish between _internal_ and _external_ faithfulness. Both measure the consistency of agent actions with their 
beliefs. The first, measures the faithfulness of an agent's internal notes and the directly following external message. 
Recall, the standard protocol alternates between observing the negotiation history so far, then generating:
1. an internal note, completing with the agent's current 'accepted offers'
2. followed by an external message directly addressing the other agent with an offer.

Internal faithfulness measures if the offers of (2) are faithful to the offers of (1). For example, if an agent is 
negotiating for a low price states an internal offer of 100, an external offer of >100 would be unfaithful.

External faithfulness is a little more tricky. Using the exact same history available to an agent writing a public offer,
we prompt the agent to estimate the acceptable offers of the other agent. Continuing our example from before, if the 
agent believes the other agent would be satisfied with an offer of 100, and external offer of >100 would be unfaithful.

### Instruction-Following
Our initial implementation measures only two types of instruction following:
1. Generation length restrictions: limits on the amount of words that can be used to generate a note or message
2. Format restrictions: agents are instructed to format internal note offers using json/dictionaries

Of course, many other instruction-following metrics could be added here.

### Miscellaneous
Other metrics that were briefly explored are:
1. Visibility: how much do agents gain from making certain hidden properties visible, e.g., opponent payoff matrices
2. Agent Description: as described in the debias section above, we looked into explicitly stating an agent's ability
