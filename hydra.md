### Terminology
---
The argument passed to the @hydra.main function <code>config_path</code> specifies the directory dedicated for config files/groups (the yaml files) – the path is relative to the working directory).

<code>Input Configs</code> are the basic building blocks from which the Output Configs (the configuration file passed to the application) are constructed. 

<code>Config files</code> (<name>.yaml files) are a form of input configs.
    
<code>Primary Config</code> is the input config (file) specified in the @hidra.main() decorator.
    
The <code>Defaults Lists</code> (keyword defaults + colon + dashes syntax) are used to compose the final config object.

### Composition order
---
If <code>_self_</code> is the first item in the Defaults List, the configs from the Defaults List are overriding parameters specified in the specific config (outside of the Defaults List). If <code>_self_</code> is the last item in the Defaults List the parameters specified in the specific config override any parameters in the Defaults List.
    
### Hydra supports custom interpolation functions and relative interpolation
---
Example:
```import math
from omegaconf import OmegaConf

def add_args(*args):
    return sum(float(x) for x in args)

def add_args_int(*args):
    return int(sum(float(x) for x in args))

def multiply_args(*args):
    return math.prod(float(x) for x in args)

def mult_args_int(*args):
    return int(math.prod(float(x) for x in args))

OmegaConf.register_new_resolver("add", add_args)
OmegaConf.register_new_resolver("mult", multiply_args)

OmegaConf.register_new_resolver("add_int", add_args_int)
OmegaConf.register_new_resolver("mult_int", mult_args_int)
```
**!!!Be mindfull of spaces in the values, they make a difference!!!**

Example usage:

```'key3': '${mult_int: ${key1}, ${key2}}```

Relative interpolation example:

```
x: 10
b:
  y: 20
  a: {x}    # 10, absolute interpolation
  b: ${.y}  # 20, relative interpolation
  c: ${..x} # 10, relative interpolation
```
To go two layers up use ${...x} etc.

---
---
### Verifying config correctness before running a job
---
```python run.py --cfg job --resolve```

---
---
### Enabling tabcompletion
Check your default shell by running <code>echo $SHELL</code>.

If it is bash, simply run <code>eval "$(python run.py -sc install=bash)"</code>.

See the instructions [here](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) if your shell is ZSH.

---
---
### Building a hydra config without changing working dir or configuring the logging system
```
import hydra
import os
from omegaconf import OmegaConf

config_path = "../.hydra"
config_name = "config.yaml"

with hydra.initialize(config_path=config_path, job_name="test_app"):
    cfg = hydra.compose(config_name=config_name, overrides=["data_dir=../data", "work_dir=../"]) #overrides=["db=mysql", "db.user=me"])
print(OmegaConf.to_yaml(cfg))
```

---
---
### Loading an Omega config
```
config_path = os.path.join(_dir, ".hydra")
config_name="config.yaml"

config = OmegaConf.load(os.path.join(config_path, config_name))
```