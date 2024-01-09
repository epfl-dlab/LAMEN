import glob 
import sys
sys.path.append('src/')

from utils import unpack_nested_yaml, get_inference_root_overrides, load_hydra_config, fill_defaults
from omegaconf import open_dict
from omegaconf.errors import MissingMandatoryValue
import os
import shutil 

def copy_file(src, dst):
    """
    Copies a file or directory from the source path to the destination path.

    :param src: The path to the source file or directory.
    :param dst: The path to the destination directory.
    """
    try:
        # Check if the source is a file or a directory
        if os.path.isfile(src):
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            # Copy the file
            shutil.copy2(src, dst)
            print(f"File copied successfully from {src} to {dst}")
        else:
            print(f"The source {src} does not exist.")
    except FileExistsError:
        print(f"The destination {dst} already exists.")
    except PermissionError:
        print(f"Permission denied when copying to {dst}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
LOGS_PATH = 'logs_copy'
folders = ['self_play', 'cross_play']

we_care = ['negotiations.csv', 'processed_negotiation.csv','interrogation.csv', '.hydra']


def process_run(run, cross_play=True):
    # try:
    print(run)
    run_time = run.split('/')[-1]
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
        
    model_names = sorted([cfg.experiments.agent_1.model_name, cfg.experiments.agent_2.model_name])
    path_names = '_'.join(model_names)
    self_play_name = cfg.experiments.agent_1.model_name

    nego_path = os.path.join(run, "negotiations.csv")
    proc_nego_path = os.path.join(run, "processed_negotiation.csv")
    inter_path = os.path.join(run, "interrogation.csv")
    hydra_path = os.path.join(run, ".hydra/")

    nego_exits = os.path.exists(nego_path)
    proc_nego_exits = os.path.exists(proc_nego_path)
    hydra_exists = os.path.exists(hydra_path)
    inter_exists = os.path.exists(inter_path)

    if cross_play:
        save_path = f'public_logs/cross_play/{path_names}/{run_time}/'

        if not nego_exits:
            return None 
        
        copy_file(nego_path, save_path)
        if proc_nego_exits:
            copy_file(proc_nego_path, save_path)
            
        if hydra_exists:
            copy_file(os.path.join(hydra_path, 'config.yaml'), os.path.join(save_path, '.hydra/config.yaml'))
            copy_file(os.path.join(hydra_path, 'hydra.yaml'), os.path.join(save_path, '.hydra/hydra.yaml'))
            copy_file(os.path.join(hydra_path, 'overrides.yaml'), os.path.join(save_path, '.hydra/overrides.yaml'))
    
    elif not cross_play:
        save_path = f'public_logs/self_play/{self_play_name}/{run_time}/'

        if not nego_exits:
            return None 
        
        copy_file(nego_path, save_path)
        if proc_nego_exits:
            copy_file(proc_nego_path, save_path)
            
        if hydra_exists:
            copy_file(os.path.join(hydra_path, 'config.yaml'), os.path.join(save_path, '.hydra/'))
            copy_file(os.path.join(hydra_path, 'hydra.yaml'), os.path.join(save_path, '.hydra/'))
            copy_file(os.path.join(hydra_path, 'overrides.yaml'), os.path.join(save_path, '.hydra/'))

        if inter_exists:            
            copy_file(inter_path, save_path)
            
    
def main(run_name='cross_play'):
    if run_name == 'cross_play':
        signal = True 
    else: 
        signal = False 
        
    runs = glob.glob(f"logs/{run_name}/runs/*")
    runs = sorted(runs)

    for run in runs: 
        runs_split = run.split('/')[-1]
        if runs_split[0] == '.':
            continue
        process_run(run, cross_play=signal)

if __name__ == "__main__":
    main()
