import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from omegaconf.errors import MissingMandatoryValue
from protocols import NegotiationProtocol, InterrogationProtocol
import glob
import os 
from utils import load_hydra_config
from copy import copy

@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    # allow updating omegaconf.dict https://stackoverflow.com/a/66296809/3723434
    # with open_dict(cfg['experiments']):
    #     _ = unpack_nested_yaml(cfg['experiments'])
    # print(OmegaConf.to_yaml(y))
    runs = glob.glob("logs/inference/runs/*")
    runs = [k for k in runs if k >= "logs/inference/runs/2023-08-21_13-35"]
    runs = ["logs/inference/runs/2023-08-29_16-45-45"]
    
    for run in runs:
        try:
            cfg = load_hydra_config(os.path.join("..", run, ".hydra/"))
        except (FileNotFoundError, MissingMandatoryValue) as e:
            print(f"{e}")
            continue

        run = os.path.join("/scratch/venia/socialgpt/GPTeam", run)

        # with open_dict(cfg['experiments']):
        #     _ = unpack_nested_yaml(cfg['experiments'])
        if os.path.exists(os.path.join(run, "negotiations.csv")):
            transcript_path = os.path.join(run, "negotiations.csv")
            game = instantiate(cfg.experiments.game)
            agent_1 = instantiate(cfg.experiments.agent_1)
            agent_2 = instantiate(cfg.experiments.agent_2)
            negotiation_protocol = NegotiationProtocol(game=game,
                                                    agent_1=agent_1,
                                                    agent_2=agent_2,
                                                    transcript=transcript_path,
                                                    **cfg.experiments.negotiation_protocol) 
            if not os.path.exists(os.path.join(run, "processed_negotiation.csv")):
                negotiation_protocol.run()
                
            cfg.experiments.negotiation_protocol.save_folder=run

            interrogation_protocol = InterrogationProtocol(questions=cfg.interrogations.questions, style="final_round", game=game,
                                                        agent_1=agent_1,
                                                        agent_2=agent_2, **cfg.experiments.negotiation_protocol)

            interrogation_protocol.run()

            time.sleep(5)



if __name__ == "__main__":
    main()
