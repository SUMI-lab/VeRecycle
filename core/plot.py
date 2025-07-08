# ================================================================================
# This code is adapted from https://github.com/LAVA-LAB/neural_stochastic_control.
# ================================================================================

from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker, colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, LogLocator, LogFormatter, LogFormatterSciNotation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from core.buffer import define_grid
from core.commons import MultiRectangularSet, RectangularSet


def plot_space(ax, spaces, plot_dimensions, fill, color, label_latex, label_plain, l_size, latex, labels):
    multi_space = MultiRectangularSet([spaces]) if isinstance(spaces, RectangularSet) else spaces
    for space in multi_space.sets:
        width, height = (space.high - space.low)[plot_dimensions]
        ax.add_patch(Rectangle(space.low[plot_dimensions], width, height, fill=fill, edgecolor=color, facecolor=color))

        mid = (space.high + space.low)[plot_dimensions] / 2
        if labels:
            text = rf'$\mathbb{{X}}_{label_latex}$' if latex else f'X_{label_plain}'
            ax.annotate(text, mid, color=color, fontsize=l_size, ha='center', va='center')


def draw_task_sets(env, ax, plot_dimensions=[0, 1], labels=False, latex=False, size=12, fill=False):
    ''' Plot the target, initial, and unsafe state sets '''

    lsize = size + 4
    # Plot spaces
    plot_space(ax, env.target_space, plot_dimensions, fill, '#30dd3a', r'\star', 'T', lsize, latex, labels)
    plot_space(ax, env.unsafe_space, plot_dimensions, fill, 'brown', r'\oslash', 'U', lsize, latex, labels)
    plot_space(ax, env.init_space, plot_dimensions, fill, 'gold', r'0', '0', lsize, latex, labels)

    return


def draw_task_decomposition(g, model, folder, name='task_decomposition'):
    ax = plt.figure().add_subplot()
    ax.set_aspect('equal', adjustable='box')

    # Calculate positions of nodes
    pos = {
        node: [
            (data.get('region', None)[0][0][0] + data.get('region', None)[0][1][0]) / 2,
            (data.get('region', None)[0][0][1] + data.get('region', None)[0][1][1]) / 2,
        ]
        for node, data in g.nodes(data=True)
    }

    selected_nodes = [
        'gold' if n == g.get_initial_node()
        else '#00FF00' if n == g.get_target_node()
        else 'white' if n == 6
        else 'black'
        for n in g.nodes(data=False)
    ]
    selected_edges = ['black' for _ in g.edges(data=False)]

    options = {
        'node_color': selected_nodes,
        'font_color': 'black',
        'font_size': 20,
        'font_family': 'Times New Roman',
        'node_shape': 's',
        'width': 2,
        'arrowstyle': '-|>',
        'arrowsize': 30,
        'ax': ax,
        'edge_color': selected_edges,
        'labels': {0: 's', 8: 't'}
    }

    # Draw graph
    nx.draw_networkx(g, pos, **options)
    draw_task_sets(model, ax, labels=False, fill=True)

    # Offset node labels
    offset_x, offset_y = 0.35, -0.35  # Define the x and y offset
    label_pos = {n: (x + offset_x, y + offset_y) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(g, label_pos, font_color='black', font_size=20, font_family='Times New Roman')

    low = model.state_space.low
    high = model.state_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    # Save plots
    for form in ['pdf', 'png']:
        filepath = Path(folder, name).with_suffix('.' + str(form))
        plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()


def draw_compositional_controller(compositional_controller, folder, name='compositional_control_graph',
                                  changed_set='original', compare_to_original=True, plot_intermediate=0,
                                  latex=False, overlay=True):
    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            'font.family': 'Times New Roman',
            'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{ulem}\usepackage{xcolor}'
        })
    ax = plt.figure(figsize=(3, 3)).add_subplot()

    if overlay:
        # Plot relevant state sets
        draw_task_sets(compositional_controller.model, ax, labels=False, fill=True)

        # Goal x-y limits
        low = compositional_controller.model.state_space.low
        high = compositional_controller.model.state_space.high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])

        # Plot the changed region
        if changed_set != 'original' and plot_intermediate != 1:
            spaces = compositional_controller.model.changed_spaces[changed_set]
            if type(spaces) is RectangularSet:
                spaces = MultiRectangularSet([spaces])
            for rect in spaces.sets:
                width, height = (rect.high - rect.low)[[0, 1]]
                ax.add_patch(Rectangle(rect.low[[0, 1]], width, height, fill=False, edgecolor='blue', hatch='//',
                                       linewidth=0))

    g = compositional_controller.task_decomposition
    pos = {node: [(data.get('region', None)[0][0][0] + data.get('region', None)[0][1][0]) / 2,
                  (data.get('region', None)[0][0][1] + data.get('region', None)[0][1][1]) / 2] for node, data in
           g.nodes(data=True)}

    for u, v in g.edges(data=False):
        bound = f'{compositional_controller.low_level_control[str((u, v))][changed_set]["lower_bound"]:.2f}'
        original_bound = f'{compositional_controller.low_level_control[str((u, v))]["original"]["lower_bound"]:.2f}'
        if plot_intermediate in [1,2]:
            nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): rf'{original_bound}' '\n'}, label_pos=0.45, ax=ax,
                                         font_color='black',
                                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='white'),
                                         font_size=8)
        else:
            nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): rf'{bound}' '\n'}, label_pos=0.45, ax=ax,
                                         font_color='black',
                                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='white'),
                                         font_size=8)
            if compare_to_original:
                if original_bound != bound:
                    nx.draw_networkx_edge_labels(g, pos, edge_labels={(u, v): '\n' rf'{original_bound}'}, label_pos=0.45,
                                                 ax=ax,
                                                 font_color='red',
                                                 bbox=dict(alpha=0.0),
                                                 font_size=8)

    path = compositional_controller.high_level_control[changed_set]
    path_edges = [(n1, n2) for n1, n2 in zip(path[:-1], path[1:])]
    original_path = compositional_controller.high_level_control['original']
    original_path_edges = [(n1, n2) for n1, n2 in zip(original_path[:-1], original_path[1:])]

    if plot_intermediate in [1,2]:
        nx.draw_networkx_edges(g, pos, edgelist=original_path_edges, width=3, alpha=0.4, edge_color="blue")
    elif plot_intermediate == 3:
        nx.draw_networkx_edges(g, pos, edgelist=original_path_edges, width=3, alpha=0.4, edge_color="red")
    else:
        nx.draw_networkx_edges(g, pos, edgelist=path_edges, width=3, alpha=0.4, edge_color="blue")
        if compare_to_original:
            nx.draw_networkx_edges(g, pos, edgelist=original_path_edges, width=3, alpha=0.4, edge_color="red")

    options = {
        'node_color': ['gold' if n == path[0] else
                       '#30dd3a' if n == path[-1] else
                       'white' for n in g.nodes(data=False)],
        'edgecolors': 'black',
        'font_color': 'black',
        'font_family': 'Times New Roman',
        'font_size': 10,
        'arrowstyle': '-|>',
        'arrowsize': 16,
        'ax': ax,
    }
    nx.draw_networkx(compositional_controller.task_decomposition, pos, **options)

    for form in ['pdf', 'png']:  # ['pdf', 'png']:
        filepath = Path(folder, name).with_suffix('.' + str(form))
        plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()


def plot_traces(env, policy_state, key, num_traces=10, len_traces=100, folder=False, filename=False, title=True):
    ''' Plot simulated traces under the given policy '''

    dim = env.plot_dim

    # Simulate traces
    traces = np.zeros((len_traces + 1, num_traces, len(env.state_space.low)))
    actions = np.zeros((len_traces, num_traces, len(env.action_space.low)))

    # Initialize traces
    for i in range(num_traces):

        key, subkey = jax.random.split(key)

        x = env.init_space.sample_single(subkey)
        traces[0, i] = x

        for j in range(len_traces):
            # Get state and action
            state = traces[j, i]
            action = policy_state.apply_fn(policy_state.params, state)
            actions[j, i] = action

            # Make step in environment
            traces[j + 1, i], key = env.step_noise_key(state, key, action)

    # Plot traces
    if dim == 2:
        ax = plt.figure().add_subplot()
        colors = plt.get_cmap('rainbow', num_traces)
        for i in range(num_traces):
            X = traces[:, i, 0]
            Y = traces[:, i, 1]

            plt.plot(X, Y, '-', color=colors(i), linewidth=1)
            plt.plot(X[0], Y[0], 'o', color=colors(i), markersize=8, label='Start' if i == 0 else "")  # 'o' for dot
            plt.plot(X[-1], Y[-1], 'x', color=colors(i), markersize=8, label='End' if i == 0 else "")  # 'x' for cross

        # Add a legend for the markers
        plt.legend()

        # Plot relevant state sets
        draw_task_sets(env, ax)

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])

        if title:
            ax.set_title(f"Simulated traces ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[0])
            plt.ylabel(env.variable_names[1])

    else:
        ax = plt.figure().add_subplot(projection='3d')

        for i in range(num_traces):
            plt.plot(traces[:, i, 0], traces[:, i, 1], traces[:, i, 2], 'o', color="gray", linewidth=1, markersize=1)
            plt.plot(traces[0, i, 0], traces[0, i, 1], traces[0, i, 2], 'ro')
            plt.plot(traces[-1, i, 0], traces[-1, i, 1], traces[-1, i, 2], 'bo')

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])

        if title:
            ax.set_title(f"Simulated traces ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[0])
            ax.set_ylabel(env.variable_names[1])
            ax.set_zlabel(env.variable_names[2])

    if folder and filename:
        # Save figure
        filename = 'policy_traces_' + filename
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()

    return traces


def plot_dataset(env, train_data=None, additional_data=None, folder=False, filename=False, title=True):
    ''' Plot the given samples '''

    dim = env.plot_dim
    if dim != 2:
        print(
            f">> Cannot create dataset plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    fig, ax = plt.subplots()

    # Plot data points in buffer that are not in the stabilizing set
    if train_data is not None:
        x = train_data[:, 0]
        y = train_data[:, 1]
        plt.scatter(x, y, color='black', s=0.1)

    if additional_data is not None:
        x = additional_data[:, 0]
        y = additional_data[:, 1]
        plt.scatter(x, y, color='blue', s=0.1)

    # Plot relevant state sets
    draw_task_sets(env, ax)

    # XY limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    if title:
        ax.set_title(f"Sample plot ({filename})", fontsize=10)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[0])
        plt.ylabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()

    return


def vector_plot(env, policy_state, vectors_per_dim=20, folder=False, filename=False, title=True):
    ''' Create vector plot under the given policy '''

    dim = env.state_dim
    if dim not in [2, 3]:
        print(
            f">> Cannot create vector plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    grid = define_grid(env.state_space.low, env.state_space.high, size=[vectors_per_dim] * dim)

    # Get actions
    action = policy_state.apply_fn(policy_state.params, grid)

    # Make step
    next_obs = env.vstep_deterministic(jnp.array(grid, dtype=jnp.float32), action,
                                       jnp.zeros((len(grid), env.noise_dim), dtype=jnp.int32))

    scaling = 1
    vectors_state = (next_obs - grid) * scaling

    # Plot vectors
    if dim == 2:
        ax = plt.figure().add_subplot()

        # Plot relevant state sets
        draw_task_sets(env, ax, fill=True)

        ax.quiver(grid[:, 0], grid[:, 1], vectors_state[:, 0], vectors_state[:, 1], alpha=1, color='k',
                  label='x at time t + 1', angles='xy', scale_units='xy', scale=1)
        ax.quiver(grid[:, 0], grid[:, 1], action[:, 0], action[:, 1], alpha=0.5, color='c', label='u at time t',
                  angles='xy', scale_units='xy', scale=0.1)

        ax.legend()

        if title:
            ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[0])
            plt.ylabel(env.variable_names[1])

    elif dim == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], vectors_state[:, 0], vectors_state[:, 1], vectors_state[:, 2],
                  length=0.5, normalize=False, arrow_length_ratio=0.5, alpha=1, color='k', label='x at time t + 1')
        ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], action[:, 0], action[:, 1], action[:, 2],
                  length=0.5, normalize=False, arrow_length_ratio=0.5, alpha=0.5, color='c', label='u at time t')

        plt.legend()

        ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[0])
            ax.set_ylabel(env.variable_names[1])
            ax.set_zlabel(env.variable_names[2])

    if folder and filename:
        # Save figure
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()

    return


def policy_plot(env, policy_state, vectors_per_dim=20, folder=False, filename=False, title=True):
    ''' Create vector plot under the given policy '''

    dim = env.state_dim
    if dim not in [2, 3]:
        print(
            f">> Cannot create vector plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    grid = define_grid(env.state_space.low, env.state_space.high, size=[vectors_per_dim] * dim)

    # Get actions
    action = policy_state.apply_fn(policy_state.params, grid)
    vectors = action

    # Plot vectors
    if dim == 2:
        ax = plt.figure().add_subplot()
        ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])

        # Plot relevant state sets
        draw_task_sets(env, ax)

        if title:
            ax.set_title(f"Policy vectors ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[0])
            plt.ylabel(env.variable_names[1])

    elif dim == 3:
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], vectors[:, 0], vectors[:, 1], vectors[:, 2],
                  length=0.5, normalize=False, arrow_length_ratio=0.5)

        ax.set_title(f"Policy vectors ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[0])
            ax.set_ylabel(env.variable_names[1])
            ax.set_zlabel(env.variable_names[2])

    if folder and filename:
        # Save figure
        filename = 'policy_vectors_' + filename
        for form in ['png']:  # ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()

    return


def plot_layout(env, folder=False, filename=False, title=True, latex=False, size=12):
    ''' Create layout plot under the given policy '''

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            'text.latex.preamble': r'\usepackage{amsfonts}'
        })

    dim = env.state_dim
    if dim != 2:
        print(
            f">> Cannot create layout plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # Plot relevant state sets
    draw_task_sets(env, ax, labels=False, latex=latex, size=size, fill=True)

    # Goal x-y limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    if title:
        ax.set_title(f"Reach-avoid specification ({filename})", fontsize=size)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[0], fontsize=size)
        plt.ylabel(env.variable_names[1], fontsize=size)
    elif latex:
        plt.xlabel('$x_1$', fontsize=size)
        plt.ylabel('$x_2$', fontsize=size)
    else:
        plt.xlabel('x1', fontsize=size)
        plt.ylabel('x2', fontsize=size)

    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)

    if folder and filename:
        # Save figure
        for form in ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    plt.close()

    return


def plot_certificate_2D(env, cert_state, folder=False, filename=False, logscale=True, title=True, labels=False,
                        latex=False, fontsize=10, num_levels=500, prob_threshold=None, changed_set=None,
                        new_threshold=None):
    ''' Plot the given certificate as a heatmap '''

    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            'text.latex.preamble': r'\usepackage{amsfonts}'
        })

    dim = env.state_dim
    fig, ax = plt.subplots()
    samples_per_dim = 101
    grid = define_grid(env.state_space.low, env.state_space.high, size=[samples_per_dim] * dim)
    grid_size = env.state_space.high - env.state_space.low

    x = np.round(grid[:, 0], 3)
    y = np.round(grid[:, 1], 3)
    out = cert_state.apply_fn(cert_state.params, grid).flatten()

    data = pd.DataFrame(data={'x': x, 'y': y, 'z': out})
    data = data.pivot(index='y', columns='x', values='z')[::-1]
    levels = MaxNLocator(nbins=num_levels).tick_values(data.min(axis=None), data.max(axis=None))
    cmap = plt.colormaps['magma']
    if logscale:
        cf = ax.contourf(data.columns, data.index, data,
                         np.logspace(np.log10(data.min(axis=None)), np.log10(data.max(axis=None)), num_levels),
                         locator=ticker.LogLocator(),
                         cmap=cmap)
        cbar = fig.colorbar(cf)
        cbar.locator = ticker.LogLocator(10)
        cbar.minorticks_on()
    else:
        cf = ax.contourf(data.columns + (grid_size[0] / samples_per_dim) / 2.,
                         data.index + (grid_size[1] / samples_per_dim) / 2., data, levels=levels,
                         cmap=cmap)
        fig.colorbar(cf, ax=ax)

    for c in cf.collections:
        c.set_edgecolor("face")

    if prob_threshold is not None:
        threshold_line = ax.contour(data.columns, data.index, data, levels=[1 / (1 - prob_threshold)], colors='black',
                                    linestyles='dotted')
        ax.clabel(threshold_line, inline=True, fmt={1 / (1 - prob_threshold): f"p = {prob_threshold:.3f}"}, fontsize=10)
    draw_task_sets(env, ax, labels=labels, latex=latex)
    fig.tight_layout()

    if title:
        ax.set_title(f"Learned Certificate ({filename})", fontsize=fontsize)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[0], fontsize=fontsize)
        plt.ylabel(env.variable_names[1], fontsize=fontsize)

    if labels:
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    if folder and filename:
        # Save figure
        for form in ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def compare_certificates(env, certificates, folder=False, filename=False, logscale=True, title=True, labels=False,
                         latex=False, fontsize=10, num_levels=500, prob_thresholds=None, changed_set=None, traces=None,
                         show_subgoals=False):
    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica",
            'text.latex.preamble': r'\usepackage{amsfonts}'
        })

    fig, axes = plt.subplots(nrows=1, ncols=len(certificates), figsize=(6, 3.5), constrained_layout=True, sharey=True)
    if len(certificates) == 1:
        axes = [axes]
    global_min = float('inf')
    global_max = float('-inf')
    dim = env.state_dim
    samples_per_dim = 101

    # Precompute the grid and range
    grid = define_grid(env.state_space.low, env.state_space.high, size=[samples_per_dim] * dim)
    grid_size = env.state_space.high - env.state_space.low

    for cert_state in certificates:
        out = cert_state.apply_fn(cert_state.params, grid).flatten()
        global_min = min(global_min, out.min())
        global_max = max(global_max, out.max())

    if logscale:
        levels = np.logspace(np.log10(global_min), np.log10(global_max), num_levels)
    else:
        levels = MaxNLocator(nbins=num_levels).tick_values(global_min, global_max)

    cmap = plt.colormaps['magma']
    for cert_state, ax, prob_threshold in zip(certificates, axes, prob_thresholds):
        ax.set_aspect('equal')
        x = np.round(grid[:, 0], 3)
        y = np.round(grid[:, 1], 3)
        out = cert_state.apply_fn(cert_state.params, grid).flatten()

        data = pd.DataFrame(data={'x': x, 'y': y, 'z': out})
        data = data.pivot(index='y', columns='x', values='z')[::-1]

        if logscale:
            cf = ax.contourf(data.columns, data.index, data, levels=levels,
                             locator=ticker.LogLocator(), cmap=cmap)
        else:
            cf = ax.contourf(data.columns + (grid_size[0] / samples_per_dim) / 2.,
                             data.index + (grid_size[1] / samples_per_dim) / 2., data, levels=levels,
                             cmap=cmap)
        for c in cf.collections:
            c.set_edgecolor("face")

        threshold_lines = ax.contour(data.columns, data.index, data, levels=[1 / (1 - p) for p in prob_threshold],
                                     colors=['black', 'black'], linestyles=['solid', 'dashed'], linewidths=1)

        cb = ax.clabel(threshold_lines, inline=True, fontsize=10, inline_spacing=10,
                       fmt={1 / (1 - p): rf"${p:.2f}$" for p in prob_threshold},
                       rightside_up=True)

        if show_subgoals:
            for x in range(3):
                for y in range(3):
                    # color = '#DAA520' if x == 0 and y == 0 else 'green' if x == 2 and y == 2 else 'gray'
                    color = 'gold' if x == 1 and y == 1 else '#30dd3a' if x == 2 and y == 1 else '#606060'

                    ax.add_patch(
                        Rectangle((x + 0.4, y + 0.4), 0.2, 0.2, fill=False, linestyle=':', ec=color))
                    num = y * 3 + x
                    text = rf'${num}$' if latex else num
                    ax.annotate(text, (x + 0.37, y + 0.6), fontsize=8, ha='right', va='top', color=color)

        draw_task_sets(env, ax, labels=labels, latex=latex)
        if changed_set is not None:
            width, height = (changed_set.high - changed_set.low)[[0, 1]]
            ax.add_patch(Rectangle(changed_set.low[[0, 1]], width, height, fill=False, edgecolor='blue', hatch='//',
                                   linewidth=0))
            # mid = (changed_set.high + changed_set.low)[[0, 1]] / 2
            # text = rf'$\mathcal{{X}}_d$' if latex else f'X_d'
            # t = ax.annotate(text, mid, color='white', fontsize=12, ha='center', va='center')
            # t.set_bbox(dict(facecolor='blue', alpha=1, edgecolor='blue'))

    cbar = fig.colorbar(cf, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)

    if logscale:
        cbar.locator = ticker.LogLocator(10)
        cbar.minorticks_on()

    if labels:
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    if traces is not None:
        for i in range(len(traces[0]) - 1):
            X = traces[:, i, 0]
            Y = traces[:, i, 1]

            plt.plot(X, Y, '-', color='white', linewidth=0.5)
            plt.plot(X[0], Y[0], 'o', color='white', markersize=1, label='Start' if i == 0 else "")  # 'o' for dot
            plt.plot(X[-1], Y[-1], 'x', color='white', markersize=2, label='End' if i == 0 else "")  # 'x' for cross

    for axis in axes:
        axis.axis('off')

    if folder and filename:
        # Save figure
        for form in ['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.' + str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()


def plot_ppo_result(agent_state, env, logger_path):
    fig, ax = plt.subplots()
    vectors_per_dim = 10
    sizes = [vectors_per_dim] * 2 + [1] * (len(env.state_space.low) - 2)
    grid = define_grid(env.state_space.low, env.state_space.high, size=sizes)
    # Get actions
    action, _ = agent_state.actor_fn(agent_state.params['actor'], grid)
    key = jax.random.split(jax.random.PRNGKey(args.seed), len(grid))
    # Make step
    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
        = env.vstep(jnp.array(grid, dtype=jnp.float32), key, action, jnp.zeros(len(grid), dtype=jnp.int32))
    scaling = 1
    vectors = (next_obs - grid) * scaling
    # Plot vectors
    ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])
    # Save figure
    filepath = Path(logger_path, 'ppo_vector.png')
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.close()


def plot_latest_ppo_traces(agent_state, args, cwd, env, env_key, global_step, verbose):
    fig, ax = plt.subplots()
    len_traces = 100
    num_traces = min(10, args.num_envs)
    next_obs, env_key, steps_since_reset = env.vreset(env_key)
    next_obs = np.array(next_obs)
    next_done = np.zeros(args.num_envs)
    obs_plot = np.zeros((len_traces, args.num_envs) + env.state_space.gymspace.shape)
    action_hist = np.zeros((len_traces, args.num_envs) + env.action_space.shape)
    for step in range(0, len_traces):
        global_step += args.num_envs
        obs_plot[step] = next_obs

        # Get action
        action, _ = agent_state.actor_fn(agent_state.params['actor'], next_obs)

        action_hist[step] = action

        next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
            = env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)
        next_done = np.logical_or(terminated, truncated)

        next_obs, next_done = np.array(next_obs), np.array(next_done)
    fig, ax = plt.subplots()
    for i in range(num_traces):
        X = obs_plot[:, i, 0]
        Y = obs_plot[:, i, 1]

        plt.plot(X, Y, '-', color="blue", linewidth=1)

        if verbose:
            print('Trace', i)
            print(obs_plot[:, i, :])
            print('With actions')
            print(action_hist[:, i, :])
            print('\n====\n')
    # Goal x-y limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[0] - 0.1, high[0] + 0.1)
    ax.set_ylim(low[1] - 0.1, high[1] + 0.1)
    ax.set_title("Simulated traces under given controller", fontsize=10)
    filepath = Path(cwd, 'logger', 'latest_ppo_traces.png')
    plt.savefig(filepath, format='png', bbox_inches='tight')
    plt.close()
