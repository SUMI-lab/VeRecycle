# ================================================================================
# This code is adapted from https://github.com/LAVA-LAB/neural_stochastic_control.
# ================================================================================

from functools import partial
from pathlib import Path
from typing import Callable, Tuple
from typing import Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from flax import struct
from flax.training.train_state import TrainState

from core.commons import args2dict


def initialize_networks(checkpoint_path, env):
    Policy_config = load_policy_config(checkpoint_path, key='Policy_config')
    V_config = load_policy_config(checkpoint_path, key='V_config')
    general_config = load_policy_config(checkpoint_path, key='general_config')

    V_neurons_withOut = V_config['neurons_per_layer']
    V_act_fn_withOut_txt = V_config['activation_fn']
    V_act_fn_withOut = orbax_parse_activation_fn(V_act_fn_withOut_txt)
    pi_neurons_withOut = Policy_config['neurons_per_layer']

    V_state, Policy_state, Policy_config, policy_neurons_withOut = create_nn_states(env, Policy_config,
                                                                                    V_neurons_withOut,
                                                                                    V_act_fn_withOut,
                                                                                    pi_neurons_withOut)
    return general_config, Policy_config, V_config, Policy_state, V_state


def load_policy_and_certificate(checkpoint_path, env):
    general_config, Policy_config, V_config, Policy_state, V_state = initialize_networks(checkpoint_path, env)

    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'general_config': general_config,
              'V_state': V_state,
              'Policy_state': Policy_state,
              'V_config': V_config,
              'Policy_config': Policy_config}
    checkpoint = orbax_checkpointer.restore(checkpoint_path, item=target)

    return (checkpoint['general_config'],
            checkpoint['Policy_config'],
            checkpoint['V_config'],
            checkpoint['Policy_state'],
            checkpoint['V_state'])


def load_policy_only(checkpoint_path, env):
    general_config, Policy_config, V_config, Policy_state, V_state = initialize_networks(checkpoint_path, env)
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'general_config': general_config,
              'Policy_state': Policy_state,
              'Policy_config': Policy_config}
    checkpoint = orbax_checkpointer.restore(checkpoint_path, item=target)
    return (checkpoint['general_config'],
            checkpoint['Policy_config'],
            V_config,
            checkpoint['Policy_state'],
            V_state)


def create_certificate_checkpoint(checkpoint_path, env, args):
    # Load policy configuration and
    Policy_config = load_policy_config(checkpoint_path, key='config')

    pi_neurons_per_layer = [args.neurons_per_layer for _ in range(args.hidden_layers)]
    V_neurons_withOut = [args.neurons_per_layer for _ in range(args.hidden_layers)] + [1]
    V_act_fn_withOut = [nn.relu for _ in range(args.hidden_layers)] + [nn.softplus]
    V_act_fn_withOut_txt = ['relu' for _ in range(args.hidden_layers)] + ['softplus']
    V_state, Policy_state, Policy_config, Policy_neurons_withOut = create_nn_states(env, Policy_config,
                                                                                    V_neurons_withOut,
                                                                                    V_act_fn_withOut,
                                                                                    pi_neurons_per_layer,
                                                                                    Policy_lr=args.Policy_learning_rate,
                                                                                    V_lr=args.V_learning_rate)

    V_config = orbax_set_config(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                seed=args.seed, RL_method=args.pretrain_method,
                                total_steps=args.pretrain_total_steps,
                                neurons_per_layer=V_neurons_withOut,
                                activation_fn_txt=V_act_fn_withOut_txt)

    general_config = args2dict(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                               seed=args.seed, probability_bound=args.probability_bound)

    # Restore state of policy network
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    target = {'model': Policy_state, 'config': Policy_config}
    Policy_state = orbax_checkpointer.restore(checkpoint_path, item=target)['model']

    return (general_config,
            Policy_config,
            V_config,
            Policy_state,
            V_state)


def export_checkpoint(logger_folder, general_config, Policy_config, V_config, Policy_state, V_state):
    ckpt = {'general_config': general_config,
            'V_state': V_state,
            'Policy_state': Policy_state,
            'V_config': V_config,
            'Policy_config': Policy_config}
    final_ckpt_path = Path(logger_folder, 'final_ckpt')
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    orbax_checkpointer.save(final_ckpt_path, ckpt,
                            save_args=flax.training.orbax_utils.save_args_from_target(ckpt),
                            force=True)
    return final_ckpt_path


vsplit_fun = jax.vmap(jax.random.split)


def vsplit(keys):
    return vsplit_fun(keys)


def load_policy_config(checkpoint_path, key):
    # First read only the config from the orbax checkpoint
    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    ckpt_restored = orbax_checkpointer.restore(checkpoint_path)
    Policy_config = ckpt_restored[key]

    return Policy_config


def create_nn_states(env, Policy_config, V_neurons_withOut, V_act_fn_withOut, pi_neurons_per_layer,
                     Policy_lr=5e-5, V_lr=5e-4):
    '''
    Create Jax state objects (for both policy and certificate)

    :param env: Benchmark model
    :param Policy_config: Configuration for policy
    :param V_neurons_withOut: Number of neurons per layer of certificate network (including the output dimension)
    :param V_act_fn_withOut: Activation function per layer of certificate network (including the output act. func.)
    :param pi_neurons_per_layer: Number of neurons per layer of policy network (including the output dimension)
    :return: V_state, Policy_state, Policy_config
    '''

    # Initialize certificate network
    certificate_model = MLP(V_neurons_withOut, V_act_fn_withOut)
    V_state = create_train_state(
        model=certificate_model,
        act_funcs=V_act_fn_withOut,
        rng=jax.random.PRNGKey(1),
        in_dim=env.state_dim,
        learning_rate=V_lr,
    )

    # Parse policy activation functions (txt -> jax functions)
    Policy_act_fn_withOut = orbax_parse_activation_fn(Policy_config['activation_fn'])
    Policy_neurons_withOut = pi_neurons_per_layer + [len(env.action_space.low)]

    # Create policy state object
    policy_model = MLP(Policy_neurons_withOut, Policy_act_fn_withOut)
    Policy_state = create_train_state(
        model=policy_model,
        act_funcs=Policy_act_fn_withOut,
        rng=jax.random.PRNGKey(1),
        in_dim=env.state_dim,
        learning_rate=Policy_lr,
    )

    return V_state, Policy_state, Policy_config, Policy_neurons_withOut


def orbax_set_config(start_datetime=None, env_name=None, layout=None, seed=None, RL_method=None, total_steps=None,
                     neurons_per_layer=None, activation_fn_txt=None):
    config = {
        'date_created': start_datetime,
        'env_name': env_name,
        'layout': layout,
        'seed': seed,
        'algorithm': RL_method,
        'total_steps': total_steps,
        'neurons_per_layer': neurons_per_layer,
        'activation_fn': activation_fn_txt
    }

    return config


def orbax_parse_activation_fn(activation_fn_txt):
    activation_fn = [None] * len(activation_fn_txt)
    for i, fn in enumerate(activation_fn_txt):
        if fn == 'relu':
            activation_fn[i] = nn.relu
        elif fn == 'tanh':
            activation_fn[i] = nn.tanh
        elif fn == 'softplus':
            activation_fn[i] = nn.softplus
        elif fn == 'None':
            activation_fn[i] = None
        else:
            print(f'(!!!) Warning: unknown activation function ({fn}) in checkpoint config encountered')

    return activation_fn


def create_batches(data_length, batch_size):
    '''
    Create batches for the given data and batch size. Returns the start and end indices to iterate over.
    :param data:
    :param batch_size:
    :return:
    '''

    num_batches = np.ceil(data_length / batch_size).astype(int)
    starts = np.arange(num_batches) * batch_size
    ends = np.minimum(starts + batch_size, data_length)

    return starts, ends


def apply_ibp_rectangular(act_fns, params, mean, radius):
    '''
    Implementation of the interval bound propagation (IBP) method from https://arxiv.org/abs/1810.12715.
    We use IBP to compute upper and lower bounds for (hyper)rectangular input sets.

    This function returns the same result as jax_verify.interval_bound_propagation(apply_fn, initial_bounds). However,
    the jax_verify version is generally slower, because it is written to handle more general neural networks.

    :param act_fns: List of flax.nn activation functions.
    :param params: Parameter dictionary of the network.
    :param mean: 2d array, with each row being an input point of dimension n.
    :param radius: 1d array, specifying the radius of the input in every dimension.
    :return: lb and ub (both 2d arrays of the same shape as `mean`
    '''

    # Broadcast radius to match shape of the mean numpy array
    radius = jnp.broadcast_to(radius, mean.shape)

    # Enumerate over the layers of the network
    for i, act_fn in enumerate(act_fns):
        layer = 'Dense_' + str(i)

        # Compute mean and radius after the current fully connected layer
        mean = mean @ params['params'][layer]['kernel'] + params['params'][layer]['bias']
        radius = radius @ jnp.abs(params['params'][layer]['kernel'])

        # Then, apply the activation function and determine the lower and upper bounds
        lb = act_fn(mean - radius)
        ub = act_fn(mean + radius)

        # Use these upper bounds to determine the mean and radius after the layer
        mean = (ub + lb) / 2
        radius = (ub - lb) / 2

    return lb, ub


class AgentState(TrainState):
    # Setting default values for agent functions to make TrainState work in jitted function
    ibp_fn: Callable = struct.field(pytree_node=False)


def create_train_state(model, act_funcs, rng, in_dim, learning_rate=0.01, ema=0, params=None):
    if params is None:
        params = model.init(rng, jnp.ones([1, in_dim]))
    else:
        params = params

    tx = optax.adam(learning_rate)
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return AgentState.create(apply_fn=jax.jit(model.apply), params=params, tx=tx,
                             ibp_fn=jax.jit(partial(apply_ibp_rectangular, act_funcs)))


@partial(jax.jit, static_argnums=(1, 2, 3,))
def lipschitz_coeff(params, weighted, CPLip, Linfty):
    '''
    Function to compute Lipschitz constants using the techniques presented in the paper.

    :param params: Neural network parameters
    :param weighted: If true, use weighted norms
    :param CPLip: If true, use the average activation operators (cplip) improvement
    :param Linfty: If true, use Linfty norm; If false, use L1 norm (currently only L1 norm is used)
    :return:
    '''

    if Linfty:
        axis = 0
    else:
        axis = 1

    minweight = jnp.float32(1e-6)
    maxweight = jnp.float32(1e6)

    if (not weighted and not CPLip):
        L = jnp.float32(1)
        # Compute Lipschitz coefficient by iterating through layers
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                L *= jnp.max(jnp.sum(jnp.abs(layer["kernel"]), axis=axis))

    elif (not weighted and CPLip):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        nmatrices = len(matrices)
        products = [matrices]
        prodnorms = [[jnp.max(jnp.sum(jnp.abs(mat), axis=axis)) for mat in matrices]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.sum(jnp.abs(mat), axis=1)) for mat in prod_list])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # interpret idx as binary number of length nmatrices - 1,
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:  # last one always true
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs


    elif (weighted and not CPLip and not Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))
            L *= lip

    elif (weighted and not CPLip and Linfty):
        L = jnp.float32(1)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
            L *= lip

    elif (weighted and CPLip and not Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])
        matrices.reverse()

        weights = [jnp.ones(jnp.shape(matrices[0])[1])]
        for mat in matrices:
            colsums = jnp.sum(jnp.multiply(jnp.abs(mat), weights[-1][jnp.newaxis, :]), axis=1)
            lip = jnp.maximum(jnp.max(colsums), minweight)
            weights.append(jnp.maximum(colsums / lip, minweight))

        matrices.reverse()
        nmatrices = len(matrices)
        products = [matrices]
        extra0 = []
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                weights[-(idx + 2)][jnp.newaxis, :]), axis=1),
                                           jnp.float32(1) / weights[-(idx + 1)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        weights[-(idx + nprods + 2)][jnp.newaxis, :]),
                                                           axis=1),
                                                   jnp.float32(1) / weights[-(idx + 1)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # interpret idx as binary number of length nmatrices - 1,
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:  # last one always true
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

    elif (weighted and CPLip and Linfty):
        L = jnp.float32(0)
        matrices = []
        for layer in params["params"].values():
            # Involve only the 'kernel' dictionaries of each layer in the network
            if "kernel" in layer:
                matrices.append(layer["kernel"])

        weights = [jnp.ones(jnp.shape(matrices[0])[0])]
        for mat in matrices:
            rowsums = jnp.sum(jnp.multiply(jnp.abs(mat), jnp.float32(1) / weights[-1][:, jnp.newaxis]), axis=0)
            lip = jnp.max(rowsums)
            weights.append(jnp.minimum(lip / rowsums, maxweight))
        weights.reverse()

        nmatrices = len(matrices)
        products = [matrices]
        extra0 = []
        prodnorms = [[jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(matrices[idx]),
                                                                jnp.float32(1) / weights[-(idx + 1)][:, jnp.newaxis]),
                                                   axis=0),
                                           weights[-(idx + 2)]))
                      for idx in range(nmatrices)]]
        for nprods in range(1, nmatrices):
            prod_list = []
            for idx in range(nmatrices - nprods):
                prod_list.append(jnp.matmul(products[nprods - 1][idx], matrices[idx + nprods]))
            products.append(prod_list)
            prodnorms.append([jnp.max(jnp.multiply(jnp.sum(jnp.multiply(jnp.abs(prod_list[idx]),
                                                                        jnp.float32(1) / weights[-(idx + 1)][:,
                                                                                         jnp.newaxis]), axis=0),
                                                   weights[-(idx + nprods + 2)]))
                              for idx in range(nmatrices - nprods)])

        ncombs = 1 << (nmatrices - 1)
        for idx in range(ncombs):
            # interpret idx as binary number of length nmatrices - 1,
            # where the jth bit determines whether to put a norm or a product between layers j and j+1
            jprev = 0
            Lloc = jnp.float32(1)
            for jcur in range(nmatrices):
                if idx & (1 << jcur) == 0:  # last one always true
                    Lloc *= prodnorms[jcur - jprev][jprev]
                    jprev = jcur + 1

            L += Lloc / ncombs

        weights.reverse()

    if weighted:
        return L, weights[-1]
    else:
        return L, None


class MLP(nn.Module):
    ''' Define multi-layer perception with JAX '''
    features: Sequence[int]
    activation_func: list

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for act_func, feat in zip(self.activation_func, self.features):
            if act_func is None:
                x = nn.Dense(feat)(x)
            else:
                x = act_func(nn.Dense(feat)(x))
        return x
