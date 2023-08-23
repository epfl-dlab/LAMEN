import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from omegaconf.errors import MissingMandatoryValue
from protocols import NegotiationProtocol
from utils import unpack_nested_yaml
import glob
import os 
from utils import load_hydra_config
import pandas as pd

@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    # allow updating omegaconf.dict https://stackoverflow.com/a/66296809/3723434
    # with open_dict(cfg['experiments']):
    #     _ = unpack_nested_yaml(cfg['experiments'])
    # print(OmegaConf.to_yaml(y))
    if cfg.rereun_experiments:
        runs = glob.glob("logs/inference/runs/*")
        runs = [k for k in runs if k >= "logs/inference/runs/2023-08-21_13-35"]
        
        for run in runs:
            try:
                cfgg = load_hydra_config(os.path.join("..", run, ".hydra/"))
            except (FileNotFoundError, MissingMandatoryValue) as e:
                print(f"{e}")
                continue

            run = os.path.join("/scratch/venia/socialgpt/GPTeam", run)
            print(run)
            # with open_dict(cfg['experiments']):
            #     _ = unpack_nested_yaml(cfg['experiments'])
            if (not os.path.exists(os.path.join(run, "processed_negotiation.csv"))) and (os.path.exists(os.path.join(run, "negotiations.csv"))):
                transcript_path = os.path.join(run, "negotiations.csv")
                game = instantiate(cfgg.experiments.game)
                agent_1 = instantiate(cfgg.experiments.agent_1)
                agent_2 = instantiate(cfgg.experiments.agent_2)
                negotiation_protocol = NegotiationProtocol(game=game,
                                                        agent_1=agent_1,
                                                        agent_2=agent_2,
                                                        transcript=transcript_path,
                                                        **cfgg.experiments.negotiation_protocol) 
                negotiation_protocol.run()
                negotiation_protocol.evaluate()
                time.sleep(5)
                
            elif os.path.exists(os.path.join(run, "processed_negotiation.csv")):
                transcript_path = os.path.join(run, "negotiations.csv")
                df = pd.read_csv(os.path.join(run, "processed_negotiation.csv"))
                print(df)
                if len(df[df["normalized_payoff"].isna()]) > 0:
                    game = instantiate(cfgg.experiments.game)
                    agent_1 = instantiate(cfgg.experiments.agent_1)
                    agent_2 = instantiate(cfgg.experiments.agent_2)
                    negotiation_protocol = NegotiationProtocol(game=game,
                                                            agent_1=agent_1,
                                                            agent_2=agent_2,
                                                            transcript=transcript_path,
                                                            **cfgg.experiments.negotiation_protocol) 
                    negotiation_protocol.evaluate()
                    time.sleep(5)
    else:
        game = instantiate(cfg.experiments.game)
        agent_1 = instantiate(cfg.experiments.agent_1)
        agent_2 = instantiate(cfg.experiments.agent_2)
        negotiation_protocol = NegotiationProtocol(game=game,
                                                agent_1=agent_1,
                                                agent_2=agent_2,
                                                **cfg.experiments.negotiation_protocol) 
        negotiation_protocol.run()
        negotiation_protocol.evaluate()
        time.sleep(5)


if __name__ == "__main__":
    main()
