import copy
import json
import os
import gymnasium as gym
import numpy as np

from argparse import Namespace
from pathlib import Path
from typing import Tuple, List, Self
from gymnasium import Env
from networkx import NetworkXNoPath, dijkstra_path
from jax import Array

from core.cegis import CEGIS
from core.commons import MultiRectangularSet
from core.jax_utils import load_policy_and_certificate, AgentState
from core.logger import Logger
from core.plot import draw_compositional_controller
from core.ppo_jax import run_PPO_initialization
from core.task_decomposition import TaskDecompositionGraph
from core.verecycle import run_VeRecycle
from core.verifier import Verifier


"""
Determines whether a repair method should be applied based on the spatial relation between the initial set, 
the target set, and the changed region.

The repair method is considered defined (i.e., worth applying) if:
- The changed region does **not** intersect with any initial region; and
- The changed region does **not** completely cover any target region.

Args:
    init (MultiRectangularSet): The set of initial regions for the subtask.
    target (MultiRectangularSet): The set of goal or target regions for the subtask.
    changed (MultiRectangularSet): The region where the environment has changed.

Returns:
    bool: True if the repair method is applicable (i.e., change is neither in initial nor fully over target regions), 
          False otherwise.
"""
def repair_method_is_defined(init: MultiRectangularSet, target: MultiRectangularSet, changed: MultiRectangularSet):
    change_intersects_init = False
    for rect_init in init.sets:
        for rect_changed in changed.sets:
            intersects = np.all(
                np.maximum(rect_changed.low, rect_init.low) <= np.minimum(rect_changed.high, rect_init.high))
            if intersects:
                change_intersects_init = True
                break
        if change_intersects_init:
            break

    change_is_superset_of_target = True
    for rect_target in target.sets:
        contained_by_any_changed = False
        for rect_changed in changed.sets:
            if np.all(rect_target.low >= rect_changed.low) and np.all(rect_target.high <= rect_changed.high):
                contained_by_any_changed = True
                break
        if not contained_by_any_changed:
            change_is_superset_of_target = False
            break

    return not change_intersects_init and not change_is_superset_of_target


class CompositionalControl:
    """
    Initialize the CompositionalControl object.

    Args:
        model (gym.Env): The environment in which the control task is defined.
        task_decomposition (TaskDecompositionGraph): Graph representing the decomposition of the overall task into subtasks.
        args (Namespace, optional): Namespace object containing experiment settings and parameters.
        controller_root (Path, optional): Path to directory where controller artifacts will be stored. If None, a path is constructed from args.

    Attributes:
        low_level_control (dict): Stores synthesis results (e.g., certified lower bounds and checkpoints) for each edge and change set.
        high_level_control (dict): Stores high-level plans (paths) computed for each change set.
        lower_bound (dict): Stores overall success probabilities for the high-level plan for each change set.
    """
    def __init__(self, model: gym.Env, task_decomposition: TaskDecompositionGraph, args: Namespace = None,
                 controller_root: Path = None) -> None:
        self.model = model
        self.task_decomposition = task_decomposition
        self.args = args

        if controller_root is None:
            name = f'model={args.model}_date={args.start_datetime}_{args.logger_suffix}_seed={args.seed}'
            self.controller_root = Path(args.cwd, 'logger', args.logger_prefix, name)
        else:
            self.controller_root = controller_root
        os.makedirs(self.controller_root, exist_ok=True)

        self.low_level_control = {}
        for edge in self.task_decomposition.edges:
            self.low_level_control[str(edge)] = {'original': {'lower_bound': 0, 'checkpoint': None}}

        self.high_level_control = {}
        self.lower_bound = {}

    """
    Synthesizes a policy and corresponding certificate for a given edge in the task decomposition using PPO and CEGIS.

    Args:
        edge (Tuple[int, int]): Directed edge in the task decomposition graph to synthesize control for.
        key (Array): RNG key for reproducibility in downstream tasks.
        max_cegis_iterations (int): Maximum number of iterations for each CEGIS loop.
        max_search_iterations (int): Maximum number of binary search iterations to refine the certified probability bound.
        ppo_seed (int): Seed for PPO initialization. If -1, falls back to args.seed.

    Returns:
        Array: Updated RNG key.
    """
    def learn_edge(self, edge: Tuple[int, int], key: Array, max_cegis_iterations: int = 10,
                   max_search_iterations: int = 5, ppo_seed: int = -1) -> Array:
        if not self.args.silent:
            print(f'============== Synthesis procedure for edge {edge} ==============')

        # Create a copy of the base model and set current subtask specifications
        model = copy.copy(self.model)
        init = self.task_decomposition.get_region(edge[0])
        target = self.task_decomposition.get_region(edge[1])
        model.update_task_spaces(init=init, target=target)

        # Do PPO initialization
        if ppo_seed == -1:
            ppo_seed = self.args.seed
        if not self.args.silent:
            print(f'Starting PPO initialization for edge {edge}, with random seed {ppo_seed}...')
        ppo_checkpoint_path = run_PPO_initialization(self.args, model, Path(self.controller_root,
                                                                            str(edge[0]) + '-' + str(edge[1]),
                                                                            "PPO_initialization"), ppo_seed)
        if not self.args.silent:
            print(f'Finished PPO initialization for edge {edge}.')

        # Perform a binary search on probability lower bound p
        if not self.args.silent:
            print(f'Starting binary search for certifiable lower bound for edge {edge}...')
        p = 0.5
        lb_p = 0
        hb_p = 1
        lb_p_checkpoint = None
        search_iterations = max_search_iterations
        while search_iterations > 0 and p >= 0.5:
            if not self.args.silent:
                print(f'====== CEGIS attempt for probability bound {p} ======')
            cegis = CEGIS(args=self.args, checkpoint_path=ppo_checkpoint_path, env=model, probability_bound=p,
                          write_folder=Path(self.controller_root, f'{edge[0]}-{edge[1]}', f'CEGIS_p={p}'))
            result_checkpoint_path, success, key = cegis.run(key, max_cegis_iterations)

            if success:
                if not self.args.silent:
                    print(f'Attempt succesful, certificate found for probability bound {p}.')
                lb_p = p
                lb_p_checkpoint = os.path.relpath(result_checkpoint_path, self.controller_root)
            else:
                hb_p = p
                if not self.args.silent:
                    print(f'Attempt unsuccesful.')

            p = (lb_p + hb_p) / 2
            search_iterations -= 1

        # Store policy, probability, certificate tuple for edge
        self.set_edge_control(edge, lb_p, str(lb_p_checkpoint))
        if not self.args.silent:
            print(f'Concluded binary search for certifiable lower bound for edge {edge} with '
                  f'max. probability bound {lb_p}.')

        return key

    """
    Solves the high-level planning problem given current edge-level guarantees for a specified change set.

    Args:
        changed_space (str): Identifier of the space that changed, determining which guarantees to consider.
        update (bool): If True, stores the resulting path and overall lower bound in this CompositionalControl object.

    Returns:
        Tuple[List[int], float]: A path from source to target node and its associated probability lower bound.
    """
    def solve_high_level(self, changed_space: str, update=True) -> (List[int], float):
        graph = self.task_decomposition
        source = graph.get_initial_node()
        target = graph.get_target_node()
        current_low_level_control = self.low_level_control.copy()
        weight = lambda u, v, attr: None if current_low_level_control[str((u, v))][changed_space]['lower_bound'] == 0 \
            else -np.log2(current_low_level_control[str((u, v))][changed_space]['lower_bound'])

        try:
            path = dijkstra_path(graph, source, target, weight)
            lower_bound = 1
            for i in range(0, len(path) - 1):
                lower_bound = lower_bound * self.low_level_control[str((path[i], path[i + 1]))][changed_space][
                    'lower_bound']
        except NetworkXNoPath:
            path = None
            lower_bound = 0

        if update:
            self.high_level_control[changed_space] = path
            self.lower_bound[changed_space] = lower_bound

        return path, lower_bound

    """
    Re-evaluates and potentially repairs edge-level guarantees after a change in the system.

    Args:
        change_name (str): Identifier for the change.
        key (Array): RNG key for downstream certificate learning tasks.
        logg (Logger, optional): Logger to store runtimes and recovered lower bounds.
        recompute_edge_method (str): Method used to recompute edge-level bounds. Options are:
            - "infimum": Uses VeRecycle directly.
            - "informed_check": Verifies VeRecycle's bound.
            - "binary_search": Runs CEGIS initialized with original certificate.
            - "binary_search_from_scratch": Runs CEGIS without reusing original certificate.

    Returns:
        Tuple[List[int], float], Array: Updated high-level path and RNG key.
    """
    def repair_policy(self, change_name: str, key: Array, logg: Logger = None, recompute_edge_method: str = ''):
        # Update lower bounds for all edges based on the changed region
        for edge in self.get_edges():
            print(f'Considering edge {edge} for change {change_name} with method {recompute_edge_method}...')
            init = self.task_decomposition.get_region(edge[0])
            target = self.task_decomposition.get_region(edge[1])
            changed = self.model.changed_spaces[change_name]

            if repair_method_is_defined(init, target, changed):
                # Create copy of model with changed set added to unsafe set
                model = copy.copy(self.model)
                new_unsafe = MultiRectangularSet(self.model.unsafe_space.sets + changed.sets)
                model.update_task_spaces(init=init, target=target, unsafe=new_unsafe)

                # Load certificate from checkpoint
                original_lb, checkpoint = self.get_edge_control(edge, 'original')
                _, _, _, policy_state, cert_state = load_policy_and_certificate(Path(self.controller_root, checkpoint),
                                                                                self.model)

                if recompute_edge_method == 'infimum':
                    if logg is not None:
                        logg.set_timer()
                    new_lb = run_VeRecycle(cert_state, original_lb, changed)
                    print(f'Recovered threshold from infimum: {new_lb}.')
                elif recompute_edge_method == 'informed_check':
                    correct_lb = run_VeRecycle(cert_state, original_lb, changed)
                    print(f'Recovered threshold from infimum (input for informed check): {correct_lb}.')
                    if logg is not None:
                        logg.set_timer()
                    new_lb = self.verify_threshold_only(correct_lb, cert_state, model, policy_state)
                    print(f'(Baseline 1) Re-verified VeRecycle threshold on existing certificate: {new_lb}.')
                elif recompute_edge_method == 'binary_search':
                    if logg is not None:
                        logg.set_timer()
                    key, new_lb, new_checkpoint = self.find_new_threshold(edge,
                                                                          Path(self.controller_root,
                                                                                       checkpoint),
                                                                          key, model, change_name,
                                                                          load_candidate=True)
                    print(f'(Baseline 2) Found new threshold with CEGIS from original certificate: {new_lb}.')
                    checkpoint = checkpoint if new_checkpoint is None else new_checkpoint
                elif recompute_edge_method == 'binary_search_from_scratch':
                    if logg is not None:
                        logg.set_timer()
                    key, new_lb, new_checkpoint = self.find_new_threshold(edge,
                                                                          Path(self.controller_root,
                                                                                       checkpoint),
                                                                          key, model, change_name,
                                                                          load_candidate=False)
                    print(f'(Baseline 3) Found new threshold with CEGIS from scratch: {new_lb}.')
                    checkpoint = checkpoint if new_checkpoint is None else new_checkpoint
                else:
                    recompute_edge_method = 'default'
                    if logg is not None:
                        logg.set_timer()
                    new_lb = run_VeRecycle(cert_state, original_lb, changed)
                    print(f'Recovered threshold from infimum: {new_lb}.')
                if logg is not None:
                    logg_time = logg.get_timer_value()
            else:
                print(f'Repair method not defined for edge {edge} with change {change_name}, logging -1 as runtime.')
                new_lb = 0
                checkpoint = None
                logg_time = -1

            if logg is not None:
                dict_key = f'method=[{recompute_edge_method}]_edge=[{str(edge[0])}-{str(edge[1])}]_changed=[{change_name}]'
                logg.append_time(key=dict_key, value=logg_time, export=False)
                logg.add_info(key=dict_key, value=new_lb, export=False)
            self.set_edge_control(edge, lower_bound=new_lb, checkpoint=checkpoint,
                                  change_set=change_name)

        # Solve and return high-level control
        return self.solve_high_level(change_name), key

    """
    Baseline method 1: Verifies whether a known threshold is still certifiable using the original certificate.

    Args:
        correct_threshold (float): Threshold recovered by VeRecycle.
        cert_state (AgentState): Original certificate state.
        model (Env): Updated environment including the change.
        policy_state (AgentState): Original policy state.

    Returns:
        float: Re-verified threshold (same as correct_threshold if verification succeeds, otherwise 0).
    """
    def verify_threshold_only(self, correct_threshold: float, cert_state: AgentState,
                              model: Env, policy_state: AgentState) -> float:
        verifier = Verifier(model, correct_threshold)
        verifier.partition_noise(model, self.args)
        finished, _, _, _, _, _ \
            = verifier.check_and_refine(0, model, self.args, cert_state, policy_state)

        return correct_threshold if finished else 0.0

    """
    Baseline methods 2 and 3: Searches for a new certifiable lower bound using CEGIS, with (B2) or without (B3) reusing 
    the original certificate.

    Args:
        edge (Tuple[int, int]): Edge for which to re-synthesize the certificate.
        policy_checkpoint_path (Path): Path to the original checkpoint containing policy and certificate networks.
        key (Array): RNG key for use in CEGIS.
        model (Env): Modified environment.
        change_name (str): Identifier for the change set.
        load_candidate (bool): If True, initializes CEGIS with the original certificate.

    Returns:
        Tuple[Array, float, str]: Updated RNG key, new certified threshold, and checkpoint path for the new certificate.
    """
    def find_new_threshold(self, edge: Tuple[int, int], policy_checkpoint_path: Path, key: Array, model: Env,
                           change_name: str, load_candidate: bool = False) -> Tuple[Array, float, str]:
        p = self.get_edge_control(edge)[0]
        lb_p = 0.0
        hb_p = 1.0
        lb_p_checkpoint = None
        search_iterations = self.args.binary_search_depth
        while search_iterations > 0 and lb_p < self.get_edge_control(edge)[0]:
            if not self.args.silent:
                print(f'====== CEGIS attempt for probability bound {p} ======')
            cegis = CEGIS(self.args, policy_checkpoint_path, model, p,
                          Path(self.controller_root, str(edge[0]) + '-' + str(edge[1]), change_name,
                               'CEGIS_p=' + str(p)),
                          load_certificate_state=load_candidate,
                          load_certificate_config=True)
            result_checkpoint_path, success, key = cegis.run(key, self.args.cegis_iterations)

            if success:
                if not self.args.silent:
                    print(f'Attempt successful, certificate found for probability bound {p}.')
                lb_p = p
                lb_p_checkpoint = os.path.relpath(result_checkpoint_path, self.controller_root)
            else:
                hb_p = p
                if not self.args.silent:
                    print(f'Attempt unsuccessful.')

            p = (lb_p + hb_p) / 2
            search_iterations -= 1
        return key, lb_p, lb_p_checkpoint

    """
    Stores the result of edge-level control synthesis and certification.

    Args:
        edge (Tuple[int, int]): The edge for which to store control data.
        lower_bound (float | None): Certified lower bound on success probability.
        checkpoint (str | None): Relative path to the checkpoint where the policy and certificate are stored.
        change_set (str): Key indicating the context for which the control data is valid (e.g., 'original' or a change name).
    """
    def set_edge_control(self, edge: Tuple[int, int], lower_bound: float | None, checkpoint: str | None,
                         change_set: str = 'original') -> None:
        if change_set not in self.low_level_control[str(edge)]:
            self.low_level_control[str(edge)][change_set] = {}
        self.low_level_control[str(edge)][change_set]['lower_bound'] = lower_bound
        self.low_level_control[str(edge)][change_set]['checkpoint'] = checkpoint
        return

    """
    Retrieves the certified threshold and associated checkpoint for a given edge and change set.

    Args:
        edge (Tuple[int, int]): The edge of interest.
        change_set (str): The change context ('original' by default).

    Returns:
        Tuple[float, str]: Threshold and relative checkpoint path.
    """
    def get_edge_control(self, edge: Tuple[int, int],
                         change_set: str = 'original') -> (float, str):
        return (self.low_level_control[str(edge)][change_set]['lower_bound'],
                self.low_level_control[str(edge)][change_set]['checkpoint'])

    """
    Returns the list of directed edges in the task decomposition graph.

    Returns:
        List[Tuple[int, int]]: Edges representing subtasks.
    """
    def get_edges(self):
        return self.task_decomposition.edges

    """
    Constructs a CompositionalControl instance by loading saved data from a folder.

    Args:
        controller_root (Path): Path to the folder containing controller JSON data.
        model (gym.Env): Environment associated with the controller.
        args (Namespace): Experiment arguments for the new controller instance.

    Returns:
        CompositionalControl: Reconstructed compositional controller object.
    """
    @classmethod
    def from_folder(cls, controller_root: Path, model: gym.Env, args: Namespace) -> Self:
        with open(Path(controller_root, 'compositional_controller.json'), 'r') as file:
            data = json.load(file)
        task_decomposition = TaskDecompositionGraph.from_dict(data['task_decomposition'])
        cc = CompositionalControl(model, task_decomposition, args, controller_root)
        cc.low_level_control = data['low_level_control']
        cc.high_level_control = data['high_level_control']
        cc.lower_bound = data['lower_bound']
        return cc

    """
    Exports the compositional controller's structure and data to a folder in JSON format. Optionally saves a visual graph.

    Args:
        plot (bool): If True, generates visualizations of the compositional control graph for each change set.
    """
    def export_to_folder(self, plot=True):
        if plot:
            for change in list(self.high_level_control.keys()):
                draw_compositional_controller(compositional_controller=self,
                                              folder=self.controller_root,
                                              name="compositional_control_graph_" + change,
                                              changed_set=change)
        # Convert the compositional controller to a JSON object
        data = json.dumps({
            'task_decomposition': self.task_decomposition.to_dict(),
            'low_level_control': self.low_level_control,
            'high_level_control': self.high_level_control,
            'lower_bound': self.lower_bound
        }, indent=4)
        output_path = Path(self.controller_root, "compositional_controller.json")
        with open(output_path, 'w') as json_file:
            json_file.write(data)