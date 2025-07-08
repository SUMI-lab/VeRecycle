import numpy as np
from core.buffer import mesh2cell_width, define_grid_jax    # Existing util functions for creating grid for IBP
from core.verifier import batched_forward_pass_ibp          # Existing implementation of IBP
from core.jax_utils import AgentState                       # Used only for typing


"""
Recovers a certified probability threshold from an existing certificate after a change in system dynamics.

Args:
    certificate (AgentState): The original certificate.
    original_lb (float): The certified probability threshold before the change.
    disrupted_region (MultiRectangularSet): The region where the system dynamics have changed.
    mesh_size (float, optional): The side length of the grid cells used to discretize each region. Defaults to 0.01.
    batch_size (int, optional): The number of points to process in parallel during IBP. Defaults to 1000.

Returns:
    float: A recovered certified probability threshold.
"""
def run_VeRecycle(certificate: AgentState, original_lb: float, disrupted_region: str,
                                       mesh_size: float = 0.01, batch_size=1000) -> float:

    # Width of each cell in the partition for each region
    verify_mesh_cell_widths = [mesh2cell_width(mesh_size, region.dimension, False) for region in
                               disrupted_region.sets]

    # Number of cells per dimension for each region
    num_per_dimensions = [
        np.array(np.ceil((region.high - region.low) / width), dtype=int)
        for region, width in zip(disrupted_region.sets, verify_mesh_cell_widths)
    ]

    # Create the verification grids for each region and concatenate them into one array
    all_points = np.vstack([
        np.hstack((
            define_grid_jax(
                region.low + 0.5 * width,
                region.high - 0.5 * width,
                size=num_dim
            ),
            np.full((np.prod(num_dim), 1), fill_value=width)
        ))
        for region, width, num_dim in zip(disrupted_region.sets, verify_mesh_cell_widths, num_per_dimensions)
    ])
    lb, _ = batched_forward_pass_ibp(certificate.ibp_fn, certificate.params,
                                     all_points[:, :disrupted_region.dimension],
                                     epsilon=0.5 * all_points[:, -1], out_dim=1, batch_size=batch_size)

    # Equation (4) from the paper
    new_lower_bound = max(min(1 - 1 / np.min(lb), original_lb), 0)

    return new_lower_bound