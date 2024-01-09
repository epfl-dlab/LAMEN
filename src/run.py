import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from protocols import NegotiationProtocol, InterrogationProtocol
from process_transcripts import ProcessTranscript
from utils import unpack_nested_yaml, fill_defaults, update_model_constructor_hydra, printv, \
    get_inference_root_overrides
import pickle
import os


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    """
export model_name=claude-2.1 \
export model_provider=anthropic
python src/run_scratch.py \
++experiments.agent_1.model_name=$model_name \
++experiments.agent_1.model_provider=$model_provider \
++experiments.agent_2.model_name=$model_name \
++experiments.agent_2.model_provider=$model_provider \
++offer_extraction_model_name=$model_name \
++offer_extraction_model_provider=$model_provider \
++max_rounds=1
    """
    with open_dict(cfg['experiments']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiments'])
        # check if any keys are missing and update default run-time overrides
        overrides = get_inference_root_overrides(cfg)
        _ = fill_defaults(cfg['experiments'], root_overrides=overrides)
        # unpack default yaml files (if any)
        _ = unpack_nested_yaml(cfg['experiments'])
        # update model constructors in case of model overrides
        update_model_constructor_hydra(cfg['experiments'])

    instantiated_models = {}
    game = instantiate(cfg.experiments.game)
    # instantiate agents and load models from cache.
    agent_1 = instantiate(cfg.experiments.agent_1)
    instantiated_models[agent_1.model_name] = agent_1.model
    del cfg.experiments.agent_2.model

    agent_2 = instantiate(cfg.experiments.agent_2, model={})
    agent_2.model = agent_1.model
    del cfg.experiments.negotiation_protocol.save_folder

    negotiation_protocol = NegotiationProtocol(game=game,
                                               agent_1=agent_1,
                                               agent_2=agent_2, save_folder=cfg.output_dir,
                                               **cfg.experiments.negotiation_protocol)
    negotiation_protocol.run()

    if cfg['evaluate']:
        nego_eval = ProcessTranscript(save_dir=cfg.output_dir, game=game, update=False)
        nego_eval.compute_metrics()
    if cfg['interrogate']:
        interrogation = InterrogationProtocol(save_folder=cfg.output_dir,
                                              game=game, agent_1=agent_1,
                                              agent_2=agent_2, start_agent_index=negotiation_protocol.start_agent_index,
                                              verbosity=negotiation_protocol.verbosity, **cfg.interrogations)
        interrogation.run()

    if cfg['pickle_agents']:
        with open(os.path.join(cfg.output_dir, 'agent_1.pkl'), "wb") as output_file:
            pickle.dump(agent_1, output_file)
        with open(os.path.join(cfg.output_dir, 'agent_2.pkl'), "wb") as output_file:
            pickle.dump(agent_2, output_file)

    time.sleep(cfg['experiment_sleep_time'])


if __name__ == "__main__":
    main()
