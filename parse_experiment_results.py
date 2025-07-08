# %%
import json
import os
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from core.commons import RectangularSet
from core.buffer import mesh2cell_width, define_grid_jax
from core.compositional_control import CompositionalControl
from core.jax_utils import load_policy_and_certificate
from core.verifier import batched_forward_pass_ibp
from models import NineRooms

sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'Times New Roman',
})


def read_csv(file, seed):
    if file.exists():
        df = pd.read_csv(file, index_col=0)
        df['seed'] = seed
    return df


def load_original_control_bounds(folder, seed):
    for subsubfolder in folder.iterdir():
        if subsubfolder.is_dir():
            controller_file = Path(rf"\\?\{str(subsubfolder / 'compositional_controller.json')}")
    with open(controller_file, 'r') as file:
        json_data = json.load(file)

        global_row = pd.DataFrame([{
            'info': json_data['lower_bound']['original'],
            'seed': seed
        }], index=[f"method=[none]_edge=[global]_changed=[original]"])

        edge_rows = []
        for edge, edge_data in json_data['low_level_control'].items():
            lower_bound = edge_data['original']['lower_bound']
            edge_rows.append({
                'info': lower_bound,
                'key': f"method=[none]_edge=[{edge[1]}-{edge[4]}]_changed=[original]",
                'seed': seed
            })
        edge_rows_df = pd.DataFrame(edge_rows)
        edge_rows_df.set_index('key', inplace=True)

        # Combine rows
        return pd.concat([global_row, edge_rows_df], ignore_index=False)


def load_dataframe_from_folder(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataframes_bounds = []
    dataframes_times = []

    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            seed_match = re.search(r'seed=\[(\d+)\]', subfolder.name)
            seed = int(seed_match.group(1)) if seed_match else None
            dataframes_bounds.append(read_csv(subfolder / "info.csv", seed))
            dataframes_times.append(read_csv(subfolder / "times.csv", seed))

    bounds = pd.concat(dataframes_bounds, ignore_index=False)
    times = pd.concat(dataframes_times, ignore_index=False)

    # Parse the 'key' column for additional fields
    for df in [bounds, times]:
        df['key'] = df.index.astype(str)
        df[['method', 'edge', 'change']] = df['key'].str.extract(
            r'method=\[([^\]]+)\]_edge=\[([^\]]+)\]_changed=\[([^\]]+)\]'
        )

    bounds = bounds.reset_index(drop=True).rename(columns={bounds.columns[0]: 'bound'}).drop(columns='key')
    bounds = bounds.drop_duplicates(subset=['seed', 'change', 'edge', 'method'])
    bounds = bounds[bounds['change'] != 'original']
    times = times.reset_index(drop=True).rename(columns={times.columns[0]: 'runtime'}).drop(columns='key')

    return bounds, times


def group_by_change_type(bounds, methods):
    marked_bounds = bounds.copy()
    marked_bounds.loc[:, 'change_type'] = marked_bounds['change'].isin(["Room5", "Room7"])
    marked_bounds.loc[:, 'change_type'] = marked_bounds['change_type'].map(
        {True: f'target adjacent',
         False: f'not target adjacent'})
    marked_bounds.loc[:, 'method'] = marked_bounds['method'].map(methods)

    table = marked_bounds.pivot_table(index='method', columns='change_type', values='bound', aggfunc=['mean', 'std'])
    total = marked_bounds.pivot_table(index='method', values='bound', aggfunc=['mean', 'std']).rename(
        columns={'bound': 'all'})
    table = table.join(total)
    table.columns = table.columns.swaplevel(0, 1)
    table.sort_index(axis=1, level=0, inplace=True)
    return table


def determine_tight_certificates(controller_folder='logger/ninerooms_original'):
    controller_folder = Path(os.getcwd(), controller_folder)
    tight_certificates_per_seed = {}
    for subfolder in controller_folder.iterdir():
        if subfolder.is_dir():
            seed_match = re.search(r'seed=(\d+)', subfolder.name)
            seed = int(seed_match.group(1)) if seed_match else None

            tight_certificates = []
            model = NineRooms()
            cc = CompositionalControl.from_folder(subfolder, model, None)
            for edge in cc.get_edges():
                original_lb, checkpoint = cc.get_edge_control(edge, 'original')
                _, _, _, _, certificate = load_policy_and_certificate(Path(cc.controller_root, checkpoint),
                                                                      cc.model)
                # Determine non-task rooms
                non_task_nodes = [node for node in cc.task_decomposition.nodes() if node not in edge]
                non_task_regions = []
                for node in non_task_nodes:
                    low = cc.task_decomposition.get_region(node).sets[0].low - 0.4
                    high = cc.task_decomposition.get_region(node).sets[0].high + 0.4
                    non_task_regions += [RectangularSet(low, high)]
                verify_mesh_cell_widths = [mesh2cell_width(0.01, region.dimension, False) for region in
                                           non_task_regions]

                # Find lower bound of the certificate for non-task rooms with IBP
                num_per_dimensions = [
                    np.array(np.ceil((region.high - region.low) / width), dtype=int)
                    for region, width in zip(non_task_regions, verify_mesh_cell_widths)
                ]
                all_points = np.vstack([
                    np.hstack((
                        define_grid_jax(
                            region.low + 0.5 * width,
                            region.high - 0.5 * width,
                            size=num_dim
                        ),
                        np.full((np.prod(num_dim), 1), fill_value=width)
                    ))
                    for region, width, num_dim in zip(non_task_regions, verify_mesh_cell_widths, num_per_dimensions)
                ])
                lb, _ = batched_forward_pass_ibp(certificate.ibp_fn, certificate.params,
                                                 all_points[:, :non_task_regions[0].dimension],
                                                 epsilon=0.5 * all_points[:, -1], out_dim=1, batch_size=1000)

                # If lower bound is higher than the safety value for the original probability threshold, add edge to
                # tight certificates list.
                if np.min(lb) >= 1 / (1 - original_lb):
                    tight_certificates.append(edge)
            tight_certificates_per_seed[seed] = tight_certificates
    return tight_certificates_per_seed


def group_by_certificate_quality(edge_bounds, controller_folder, methods):
    tight_certificate_list = determine_tight_certificates(controller_folder=controller_folder)
    edge_bounds.loc[:, "group"] = edge_bounds.apply(
        lambda row: tuple(map(int, row["edge"].split("-"))) in tight_certificate_list.get(row["seed"], []),
        axis=1
    )
    edge_bounds.loc[:, 'group'] = edge_bounds['group'].map({True: f'tight', False: f'loose'})
    edge_bounds.loc[:, 'method'] = edge_bounds['method'].map(methods)
    table = edge_bounds.pivot_table(index='method', columns='group', values='bound', aggfunc=['mean', 'std'])
    total = edge_bounds.pivot_table(index='method', values='bound', aggfunc=['mean', 'std']).rename(columns={'bound': 'all'})
    table = table.join(total)
    table.columns = table.columns.swaplevel(0, 1)
    table.sort_index(axis=1, level=0, inplace=True)
    return table

control_folder = Path(os.getcwd(), 'logger/ninerooms_original')
experiment_folder = Path(os.getcwd(), 'logger/experiment_1')

methods = {
    "infimum": 'VeRecycle',
    "informed_check": 'B-I',
    "binary_search": 'B-II',
    "binary_search_from_scratch": 'B-III'
}

# Load the required data from the experiment folders
bounds, times = load_dataframe_from_folder(experiment_folder)
bounds = bounds[bounds['method'].isin(methods.keys())]
times = times[times['method'].isin(methods.keys())]

# Create Table 1 (Experiment 1)
edge_bounds = bounds[bounds['edge'] != "global"]
edge_bounds = group_by_certificate_quality(edge_bounds, control_folder, methods=methods).round(2)
times = times[times['edge'] != "global"]
times = times[times["runtime"] != -1]
times_table = times.pivot_table(index='method', values='runtime', aggfunc=['mean', 'std']).round(2)
times_table.columns = times_table.columns.swaplevel(0, 1)
print("========= Experiment 1 Results =======")
print(times_table.rename(index=methods).loc[list(methods.values())].to_string())
print(edge_bounds.loc[list(methods.values())].to_string())

# Table 2 (Experiment 2)
global_bounds = bounds[bounds['edge'] == "global"]
global_table = group_by_change_type(global_bounds, methods).round(2)
print("========= Experiment 2 Results =======")
print(global_table.loc[list(methods.values())].to_string())
