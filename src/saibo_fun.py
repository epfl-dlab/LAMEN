import os

import hydra.utils

from flows import logging
from flows.utils import caching_utils
from flows.utils.caching_utils import clear_cache

caching_utils.CACHING_PARAMETERS.do_caching = True # Set to false to disable caching
# clear_cache() # Uncomment this line to clear the cache

from flows.flow_launchers import FlowLauncher

# logging.set_verbosity_debug()  # Uncomment this line to see verbose logs
from flows.base_flows import SequentialFlow

from flows import flow_verse
from flows.utils.general_helpers import read_yaml_file

from martinjosifoski.OpenAIChatAtomicFlow import OpenAIChatAtomicFlow

if __name__=="__main__":
    api_keys = {"openai": os.getenv("OPENAI_API_KEY")}

    root_dir = "."
    cfg_path = os.path.join(root_dir, "configs", "flows", "test_flow.yaml")
    overrides_config = read_yaml_file(cfg_path)

    flow1 = OpenAIChatAtomicFlow.instantiate_from_default_config(overrides=overrides_config)
    flow2 = OpenAIChatAtomicFlow.instantiate_from_default_config(overrides=overrides_config)
    flow3 = OpenAIChatAtomicFlow.instantiate_from_default_config(overrides=overrides_config)

    sf = SequentialFlow([flow1, flow2, flow3])

    _, outputs = FlowLauncher.launch(
    flow=sf,
    data={"question": "what is monkey?"},
    api_keys=api_keys,
    path_to_output_file=".",
    )




