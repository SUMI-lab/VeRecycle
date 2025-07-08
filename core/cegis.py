from pathlib import Path
from typing import Tuple

import jax
import numpy as np
from jax import Array

from core.buffer import Buffer
from core.jax_utils import (load_policy_and_certificate, load_policy_only, create_certificate_checkpoint,
                            export_checkpoint)
from core.learner import Learner
from core.plot import plot_certificate_2D, plot_traces, policy_plot, vector_plot, plot_dataset
from core.verifier import Verifier


class CEGIS:
    def __init__(self, args, checkpoint_path, env, probability_bound, write_folder,
                 load_certificate_state: bool = False, load_certificate_config: bool = False):
        self.args = args
        self.env = env
        self.probability_bound = probability_bound
        self.write_folder = write_folder

        self.check_min_lipschitz()

        # Load stored policy and initialize certificate
        if load_certificate_state:
            (self.general_config,
             self.policy_config,
             self.certificate_config,
             self.policy_state,
             self.certificate_state) = load_policy_and_certificate(checkpoint_path, env)
        elif load_certificate_config:
            (self.general_config,
             self.policy_config,
             self.certificate_config,
             self.policy_state,
             self.certificate_state) = load_policy_only(checkpoint_path, env)
        else:
            (self.general_config,
             self.policy_config,
             self.certificate_config,
             self.policy_state,
             self.certificate_state) = create_certificate_checkpoint(checkpoint_path, env, args)

        # Set learner and verifier objects
        self.learner = Learner(env, args=args)
        self.verifier = Verifier(env, probability_bound)
        self.verifier.partition_noise(env, args)

        # Define counterexample buffer
        self.counterx_buffer = Buffer(dim=env.state_space.dimension,
                                      max_size=args.num_counterexamples_in_buffer,
                                      extra_dims=4)

        self.update_policy_after_iteration = 100

    def run(self, key: Array, max_iterations: int) -> Tuple[Path | None, bool, Array]:
        finished = False
        for i in range(max_iterations):
            print(f'Iteration {i} of CEGIS attempt for probability bound {self.probability_bound}.')
            print(f'Learning...')
            key = self.learner_iteration(key, self.args.update_policy and i >= self.update_policy_after_iteration)
            print(f'Checking...')
            finished, counterx, counterx_weights, counterx_hard, _, _ \
                = self.verifier.check_and_refine(i, self.env, self.args, self.certificate_state, self.policy_state)

            if self.args.plot_intermediate:
                self.plot_intermediate(i)

            if finished or i == max_iterations - 1:
                break
            else:
                self.update_counterexamples(counterx, counterx_hard, counterx_weights)

                # Uniformly refine verification grid to smaller mesh
                self.args.mesh_verify_grid_init = np.maximum(0.75 * self.args.mesh_verify_grid_init,
                                                             self.args.mesh_verify_grid_min)

        if finished:
            final_checkpoint_path = export_checkpoint(self.write_folder,
                                                      self.general_config,
                                                      self.policy_config,
                                                      self.certificate_config,
                                                      self.policy_state,
                                                      self.certificate_state)
            plot_certificate_2D(self.env, self.certificate_state, folder=self.write_folder, filename='certificate',
                                latex=self.args.latex)
            plot_traces(self.env, self.policy_state, jax.random.PRNGKey(0), folder=self.write_folder,
                        filename='certified')
            policy_plot(self.env, self.policy_state, folder=self.write_folder, filename='certified')
            return final_checkpoint_path, finished, key
        else:
            return None, finished, key

    def plot_intermediate(self, i):
        # Plot traces
        filename = f"{self.args.start_datetime}_policy_traces_iteration={i}"
        plot_traces(self.env, self.policy_state, key=jax.random.PRNGKey(2), folder=self.write_folder,
                    filename=filename,
                    title=(not self.args.presentation_plots))
        # Plot vector of effect of policy given the dynamics
        filename = f"{self.args.start_datetime}_policy_effect_vector_plot_iteration={i}"
        vector_plot(self.env, self.policy_state, folder=self.write_folder, filename=filename,
                    title=(not self.args.presentation_plots))
        # Plot vector plot of policy
        filename = f"{self.args.start_datetime}_policy_vector_plot_iteration={i}"
        policy_plot(self.env, self.policy_state, folder=self.write_folder, filename=filename,
                    title=(not self.args.presentation_plots))
        # Plot base training samples + counterexamples
        filename = f"{self.args.start_datetime}_train_samples_iteration={i}"
        plot_dataset(self.env, additional_data=self.counterx_buffer.data, folder=self.write_folder, filename=filename,
                     title=(not self.args.presentation_plots))
        # Plot current certificate
        filename = f"{self.args.start_datetime}_certificate_iteration={i}"
        plot_certificate_2D(self.env, self.certificate_state, folder=self.write_folder, filename=filename,
                            title=(not self.args.presentation_plots),
                            labels=(not self.args.presentation_plots), latex=self.args.latex)

    def check_min_lipschitz(self):
        try:
            min_lipschitz = (1 / (1 - self.args.probability_bound) - 1) / self.env.init_unsafe_dist * (
                    self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * self.args.min_lip_policy_loss)
            if self.args.mesh_loss * min_lipschitz > 1:
                print(
                    '(!!!) Severe warning: mesh_loss is (much) too high. Impossible for loss to converge to 0 (which '
                    'likely makes it very hard to learn a proper martingale). Suggested maximum value:',
                    0.2 / min_lipschitz)
            elif self.args.mesh_loss * min_lipschitz > 0.2:
                print(
                    'Warning: mesh_loss is likely too high for good convergence of loss to 0. Suggested maximum value:',
                    0.2 / min_lipschitz)
        except:
            pass

    def update_counterexamples(self, counterx, counterx_hard, counterx_weights):
        # Append weights to the counterexamples
        counterx_plus_weights = np.hstack(
            (counterx[:, :self.env.state_dim], counterx_weights, counterx_hard.reshape(-1, 1)))
        # Add counterexamples to the counterexample buffer
        if not self.args.silent:
            print(f'\nRefresh {(self.args.counterx_refresh_fraction * 100):.1f}% of the counterexample buffer')
        self.counterx_buffer.append_and_remove(refresh_fraction=self.args.counterx_refresh_fraction,
                                               samples=counterx_plus_weights,
                                               perturb=self.args.perturb_counterexamples,
                                               cell_width=counterx[:, -1],
                                               weighted_sampling=self.args.weighted_counterexample_sampling)

    def learner_iteration(self, key, update_policy):
        num_batches = int(
            np.ceil((self.args.num_samples_per_epoch + len(self.counterx_buffer.data)) / self.args.batch_size))
        for _ in range(self.args.epochs * num_batches):
            # Main train step function: Defines one loss function for the provided batch of train data and minimizes it
            certificate_grads, policy_grads, infos, key, loss_exp_decr, samples_in_batch = self.learner.train_step(
                key=key,
                V_state=self.certificate_state,
                Policy_state=self.policy_state,
                counterexamples=self.counterx_buffer.data[:, :-1],
                mesh_loss=self.args.mesh_loss,
                probability_bound=self.probability_bound,
                expDecr_multiplier=self.args.expDecr_multiplier
            )
            if np.isnan(infos['0. total']):
                print(
                    '(!!!) Severe warning: The learned losses contained NaN values, which indicates most probably at '
                    'an error in the learner module.')
            else:
                # Update certificate and policy
                if self.args.update_certificate:
                    self.certificate_state = self.certificate_state.apply_gradients(grads=certificate_grads)
                if update_policy:
                    self.policy_state = self.policy_state.apply_gradients(grads=policy_grads)

        return key
