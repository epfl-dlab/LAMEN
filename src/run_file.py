"""
Run class to conduct experiments
"""
import fire
import os
from datetime import datetime as dt
from agents import NegotiationAgent, NegotiationProtocol
from game_utils import load_game

RESULTS_FOLDER = 'data/results'


def run_experiment(game_name='rio_copa', max_rounds=10, stop_condition='context_full', exp_name='', verbosity=1,
                   **kwargs):

    save_folder = os.path.join(RESULTS_FOLDER, exp_name, dt.now().strftime('%Ym%d_%H%M%S'))
    game = load_game(game_name=game_name)

    # TODO: init NegotiationAgents using flags
    agent_0 = NegotiationAgent()
    agent_1 = NegotiationAgent()
    negotiation = NegotiationProtocol(agents=[agent_0, agent_1], game=game, stop_condition=stop_condition,
                                      max_rounds=max_rounds, save_folder=save_folder)
    negotiation.run()


if __name__ == "__main__":
    fire.Fire(run_experiment)
