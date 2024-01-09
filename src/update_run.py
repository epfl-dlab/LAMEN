import time
import hydra
from hydra.utils import instantiate
from utils import unpack_nested_yaml, get_inference_root_overrides
from omegaconf import DictConfig, open_dict
from omegaconf.errors import MissingMandatoryValue
from protocols import NegotiationProtocol, InterrogationProtocol
from games import Game 
from agents import NegotiationAgent
import glob
import os
from utils import load_hydra_config, fill_defaults
from copy import copy
from multiprocessing import Pool


def process_run(run):
    # try:
    print(run)
    try:
        cfg = load_hydra_config(os.path.join("..", run, ".hydra/"))
    except (FileNotFoundError, MissingMandatoryValue) as e:
        print(f"{e}")
        return
    with open_dict(cfg['experiments']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiments'])
        # check if any keys are missing and update default run-time overrides
        overrides = get_inference_root_overrides(cfg)
        _ = fill_defaults(cfg['experiments'], root_overrides=overrides)
        # unpack default yaml files (if any)
        _ = unpack_nested_yaml(cfg['experiments'])

    # elif (os.path.exists(os.path.join(run, "negotiations.csv"))) & (cfg.experiments.agent_1.model_name=='gpt-4'):
    #     print(cfg.experiments.agent_1.model_name)

            # transcript_path = os.path.join(run, "negotiations.csv")
            # game = instantiate(cfg.experiments.game)

            # agent_1 = instantiate(cfg.experiments.agent_1)
            # agent_2 = instantiate(cfg.experiments.agent_2)
            # negotiation_protocol = NegotiationProtocol(game=game,
            #                                         agent_1=agent_1,
            #                                         agent_2=agent_2,
            #                                         transcript=transcript_path,
            #                                         **cfg.experiments.negotiation_protocol)
            # try: 
            #     negotiation_protocol.evaluate(update=True)  
            # except (KeyError, AttributeError) as e:
            #     print(f'EVALUATION FAILED {e}')
            
    # condition = ((cfg.experiments.agent_1.model_name=='gpt-3.5-turbo') and (cfg.experiments.agent_2.model_name=='claude-2')) or \
    #             ((cfg.experiments.agent_2.model_name=='gpt-3.5-turbo') and (cfg.experiments.agent_1.model_name=='claude-2'))
    try: 
        if (not os.path.exists(os.path.join(run, "interrogation.csv"))):
            transcript_path = os.path.join(run, "negotiations.csv")

            game = instantiate(cfg.experiments.game)

            agent_1 = instantiate(cfg.experiments.agent_1)
            agent_2 = instantiate(cfg.experiments.agent_2)
            del cfg.experiments.negotiation_protocol.save_folder 

            negotiation_protocol = InterrogationProtocol(game=game,
                                                    agent_1=agent_1,
                                                    agent_2=agent_2,
                                                    save_folder=run,
                                                    transcript=transcript_path,
                                                    **cfg.experiments.negotiation_protocol,
                                                    **cfg.interrogations)            
            negotiation_protocol.run()


    except Exception as e:
        print(f'issue with load - {e}') 
    
        # else: 
        #     game = instantiate(cfg.experiments.game)

        #     agent_1 = instantiate(cfg.experiments.agent_1)
        #     agent_2 = instantiate(cfg.experiments.agent_2)
        #     del cfg.experiments.negotiation_protocol.save_folder 

        #     negotiation_protocol = NegotiationProtocol(game=game,
        #                                             agent_1=agent_1,
        #                                             agent_2=agent_2,
        #                                             save_folder=run,
        #                                             **cfg.experiments.negotiation_protocol)
        #     try: 
        #         negotiation_protocol.run()
        #     except: 
        #         pass 
        #     negotiation_protocol.evaluate(update=False)  

    # except Exception as e:
    #     print(f'failed on run {e}')
            # negotiation_protocol.interrogate(questions=cfg.interrogations.questions,
            #                                  interrogation_style=cfg.interrogations.style)
def main(run_name='cross_play'):
    runs = glob.glob(f"public_logs/transcripts/{run_name}/claude-2_gpt-3.5-turbo/*")
    runs = sorted(runs)

    for run in runs: 
        process_run(run)

if __name__ == "__main__":
    main()
