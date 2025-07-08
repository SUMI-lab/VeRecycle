# %%
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import jax

from core.compositional_control import CompositionalControl
from core.parse_args import parse_arguments
from core.plot import plot_layout, draw_compositional_controller, draw_task_decomposition
from models.nine_rooms import NineRooms, create_task_decomposition_nine_rooms

# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

# Define argument object
args = parse_arguments(linfty=False, datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cwd=os.getcwd())
model = NineRooms(args)

key = jax.random.PRNGKey(args.seed)
key, subkey = jax.random.split(key)

if args.load_folder_compositional == '':
    tdg = create_task_decomposition_nine_rooms()
    cc = CompositionalControl(model, tdg, args=args)
else:
    load_path = Path(args.cwd, args.load_folder_compositional)
    cc = CompositionalControl.from_folder(load_path, model, args=args)

for edge in cc.get_edges():
    max_attempts = args.max_PPO_attempts
    for attempt in range(max_attempts):
        if cc.get_edge_control(edge)[0] > 0:
            cc.export_to_folder()
            break

        print(f'Starting attempt {attempt + 1} of max. {max_attempts} to learn edge {edge}...')
        key = cc.learn_edge(edge,
                            key,
                            max_cegis_iterations=args.cegis_iterations,
                            max_search_iterations=args.binary_search_depth,
                            ppo_seed=args.seed + attempt * 100)

cc.solve_high_level('original')
cc.export_to_folder()
