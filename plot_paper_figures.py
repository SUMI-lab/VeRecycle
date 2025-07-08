import jax
import numpy as np
from datetime import datetime
from pathlib import Path
from core.commons import RectangularSet
from core.compositional_control import CompositionalControl
from core.jax_utils import load_policy_and_certificate
from core.parse_args import parse_arguments
from core.plot import draw_compositional_controller, compare_certificates, plot_traces
from models.nine_rooms import NineRooms
import os


def plot_example_reclaiming_bounds(args, folder, edge, change, change_room, plot_intermediate):
    cc = CompositionalControl.from_folder(Path(args.cwd, folder), NineRooms(args), args)
    cc.model.update_task_spaces(init=cc.task_decomposition.get_region(edge[0]),
                                target=cc.task_decomposition.get_region(edge[1]),
                                unsafe=cc.model.unsafe_space)

    lb, certificate_checkpoint = cc.get_edge_control(edge, 'original')
    _, _, _, policy_state, cert_state = load_policy_and_certificate(Path(cc.controller_root, certificate_checkpoint),
                                                                    cc.model)
    lb_inf, certificate_checkpoint = cc.get_edge_control(edge, change)
    traces = plot_traces(cc.model, policy_state, jax.random.PRNGKey(1), num_traces=10, len_traces=100, folder=False,
                         filename=False, title=True)
    if plot_intermediate:
        compare_certificates(cc.model, [cert_state], folder=Path(args.cwd, 'figures'),
                             filename='reclaiming_bounds_example_1', labels=False, title=False, num_levels=1000,
                             prob_thresholds=[[lb]], changed_set=None, latex=True, traces=traces,
                             show_subgoals=True)
        compare_certificates(cc.model, [cert_state], folder=Path(args.cwd, 'figures'),
                             filename='reclaiming_bounds_example_2', labels=False, title=False, num_levels=1000,
                             prob_thresholds=[[lb]], changed_set=change_room, latex=True, traces=traces,
                             show_subgoals=True)
    compare_certificates(cc.model, [cert_state], folder=Path(args.cwd, 'figures'),
                         filename='reclaiming_bounds_example', labels=False, title=False, num_levels=1000,
                         prob_thresholds=[[lb_inf, lb]], changed_set=change_room, latex=True, traces=traces,
                         show_subgoals=True)


def plot_example_compositional(args, folder, change, change_room, plot_intermediate):
    cc = CompositionalControl.from_folder(Path(args.cwd, folder), NineRooms(args), args)
    cc.model.changed_spaces = {change: change_room}
    # for plot_intermediate, we need
    # 1. the original graph
    # 2. the original graph with changed set
    # 3. the updated probabilities and path changed to red
    # 4. the new blue path
    if plot_intermediate:
        for i in range(1,4):
            draw_compositional_controller(cc, Path(args.cwd, 'figures'), name='new_path_example_overlay_'+str(i),
                                      changed_set=change, compare_to_original=True, latex=True,
                                      plot_intermediate=i)
        draw_compositional_controller(cc, Path(args.cwd, 'figures'), name='new_path_example_overlay_4',
                                      changed_set=change, compare_to_original=True, latex=True)
    draw_compositional_controller(cc, Path(args.cwd, 'figures'), name='new_path_example_overlay',
                                  changed_set=change, compare_to_original=True, latex=True)



def plot_certificate_refinement(args, folders, edge, change, change_room):
    certificates = []
    thresholds = []
    model = NineRooms(args)
    for folder in folders:
        cc = CompositionalControl.from_folder(Path(args.cwd, folder), model, args)
        model.update_task_spaces(init=cc.task_decomposition.get_region(edge[0]),
                                 target=cc.task_decomposition.get_region(edge[1]),
                                 unsafe=model.unsafe_space)
        lb_original, _ = cc.get_edge_control(edge, 'original')
        lb_new, checkpoint = cc.get_edge_control(edge, change)
        _, _, _, _, certificate = load_policy_and_certificate(Path(cc.controller_root, checkpoint), cc.model)
        certificates.append(certificate)
        thresholds.append([lb_new, lb_original])

    compare_certificates(model, certificates, folder=Path(args.cwd, 'figures'),
                         filename='certificate_refinement_example', labels=False, title=False, num_levels=1000,
                         changed_set=change_room, latex=True, show_subgoals=False, prob_thresholds=thresholds)


args = parse_arguments(linfty=False, datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cwd=os.getcwd())
VeRecycle = ('logger/experiment_1/repair=[infimum]_seed=[1]_date=['
             '2024-12-19_12-18-19]/model=NineRooms_date=2024-11-23_14-19-45__seed=1')
baseline = ('logger/experiment_1/repair=[binary_search]_seed=[1]_date=['
            '2024-12-20_19-29-29]/model=NineRooms_date=2024-11-23_14-19-45__seed=1')
room2 = RectangularSet(low=np.array([2.0, 0.0]), high=np.array([3.0, 1.0]))

# Figure 1(a)
plot_example_reclaiming_bounds(args, folder=VeRecycle, edge=(4, 5), change="Room2", change_room=room2,
                               plot_intermediate=False)
# Figure 1(b)
plot_example_compositional(args, folder=VeRecycle, change="Room2", change_room=room2, plot_intermediate=False)
# Figure 2
plot_certificate_refinement(args, folders=[VeRecycle, baseline], edge=(4, 5), change="Room2", change_room=room2)
