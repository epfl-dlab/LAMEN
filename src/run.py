import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    print(cfg["experiments"]["issues"])
    print(cfg["experiments"]["agents"][0])
    
    # TODO initilaize the agents
    # Run through the negotiations
    # With the features from the config file.

if __name__=="__main__":
    main()