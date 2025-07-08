# %%
import os
import shutil
from datetime import datetime
from pathlib import Path
import jax
import numpy as np
import models
from core.commons import MultiRectangularSet, RectangularSet
from core.compositional_control import CompositionalControl
from core.logger import Logger
from core.parse_args import parse_arguments

# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

# Define argument object by parsing all arguments
args = parse_arguments(linfty=False, datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cwd=os.getcwd())

# Set the variables for logging the experiment
experiment_name = f'method=[{args.method}]_seed=[{args.seed}]_date=[{args.start_datetime}]'
logger_folder = Path(args.cwd, args.experiment_folder, experiment_name)
assert not os.path.exists(logger_folder), print(f'The experiment {experiment_name} already exists in the logger folder '
                                                f'{args.experiment_folder}. Either remove the existing results, '
                                                f'change the argument \'logger_folder\', or change on of the three '
                                                f'experiment parameters (method, seed, original controller.')
args.logger_folder = logger_folder
LOGG = Logger(logger_folder=logger_folder, round_decimals=6)
LOGG.export_args(args)

print(f'========= Starting repair experiment ========='
      f'\n- Starting time: {args.start_datetime}'
      f'\n- Model: {args.model}'
      # f'\n- Changes: {args.changes}'
      f'\n- Method: {args.method}'
      f'\n- Random seed: {args.seed}'
      f'\n- Original controller: {args.load_folder_compositional}'
      f'\n- Results folder: {logger_folder}'
      f'\n==============================================')

# Create a copy of the original verified compositional controller to experiment with repairs.
load_path = Path(args.cwd, args.load_folder_compositional)
assert os.path.exists(Path(load_path, 'compositional_controller.json')), print(
    "Cannot run Experiment 1 without a verified compositional controller as input.\n"
    "Set the argument \"load_folder_compositional\" to a folder containing "
    "\"compositional_controller.json\" (relative to repository root).")
LOGG.set_timer()
shutil.copytree(load_path, Path(logger_folder, load_path.name))
print(f'Created a copy of the original verified compositional controller to repair in {LOGG.get_timer_value()} '
      f'seconds.')
controller_root = Path(logger_folder, load_path.name)

# Create model and load compositional controller.
model_function = models.get_model_fun(args.model)
model = model_function(args)
cc = CompositionalControl.from_folder(controller_root, model, args=args)
key = jax.random.PRNGKey(args.seed)

model.changed_spaces = {
    # 'Room0': MultiRectangularSet([RectangularSet(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))]),
    'Room1': MultiRectangularSet([RectangularSet(low=np.array([1.0, 0.0]), high=np.array([2.0, 1.0]))]),
    'Room2': MultiRectangularSet([RectangularSet(low=np.array([2.0, 0.0]), high=np.array([3.0, 1.0]))]),
    'Room3': MultiRectangularSet([RectangularSet(low=np.array([0.0, 1.0]), high=np.array([1.0, 2.0]))]),
    'Room4': MultiRectangularSet([RectangularSet(low=np.array([1.0, 1.0]), high=np.array([2.0, 2.0]))]),
    'Room5': MultiRectangularSet([RectangularSet(low=np.array([2.0, 1.0]), high=np.array([3.0, 2.0]))]),
    'Room6': MultiRectangularSet([RectangularSet(low=np.array([0.0, 2.0]), high=np.array([1.0, 3.0]))]),
    'Room7': MultiRectangularSet([RectangularSet(low=np.array([1.0, 2.0]), high=np.array([2.0, 3.0]))]),
    # 'Room8': MultiRectangularSet([RectangularSet(low=np.array([2.0, 2.0]), high=np.array([3.0, 3.0]))])
}

for changed_set_name in model.changed_spaces.keys():
    print(f'Considering changed set "{changed_set_name}"...')
    (_, nlb), key = cc.repair_policy(change_name=changed_set_name,
                                     key=key,
                                     logg=LOGG,
                                     recompute_edge_method=args.method)
    LOGG.add_info(key=f'method=[{args.method}]_edge=[global]_changed=[{changed_set_name}]', value=nlb)
    LOGG.export_times()
    LOGG.export_info()
    cc.export_to_folder()

print(f'========= Finished repair experiment ========='
      f'\n- Finishing time: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
      f'\n- Results folder: {logger_folder}'
      f'\n==============================================')
