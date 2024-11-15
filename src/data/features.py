from copy import deepcopy
import torch


def encode_state(obses, config, my_index=0):

    """Creates 3-D state representation

    Args:
        obses: states info
        config: enviroment configuration
        my_index: index of my goose

    Returns:
        enc_state

    """
    n_rows, n_cols, n_enemies, dims = _unpack_config(config)

    enc_state = torch.ones((n_rows, n_cols)).long() * dims
    geese_info = deepcopy(obses[-1]['geese'])
    food_info = torch.tensor(obses[-1]['food'])

    my_goose = geese_info.pop(my_index)

    # my goose
    enc_state = _process_goose(my_goose, enc_state, head_val=0,
                               body_val=2 * (n_enemies + 1),
                               tail_val=n_enemies + 1)

    # enemies
    for idx, goose in enumerate(geese_info):
        enc_state = _process_goose(goose, enc_state, head_val=idx + 1,
                                   body_val=2 * (n_enemies + 1) + 1 + idx,
                                   tail_val=n_enemies + 2 + idx)

    # food
    enc_state[food_info // n_cols, food_info % n_cols] = dims - 1

    # centering
    head_pos = (enc_state == 0).nonzero(as_tuple=False)
    enc_state = _center(enc_state, head_pos)

    # one hot encoding
    enc_state = torch.nn.functional.one_hot(enc_state).permute(2, 0, 1)

    # add previous state info
    if len(obses) > 1:
        prev_enc_state = _encode_prev_state(obses, head_pos, config)
        enc_state = enc_state + prev_enc_state

    # remove redundant dim
    enc_state = enc_state[:-1, :, :]
    enc_state = enc_state.view(-1, dims, n_rows, n_cols).type(torch.float32)

    return enc_state


def _unpack_config(config):

    n_rows = config['n_rows']
    n_cols = config['n_cols']
    n_enemies = config['n_enemies']
    dims = config['n_dims']

    return n_rows, n_cols, n_enemies, dims


def _encode_prev_state(obses, head_pos, config, my_index=0):

    n_rows, n_cols, n_enemies, dims = _unpack_config(config)

    prev_enc_state = torch.ones((n_rows, n_cols)).long() * dims
    prev_geese_info = deepcopy(obses[-2]['geese'])
    my_goose = prev_geese_info.pop(my_index)

    # my goose
    prev_enc_state = _put_head(my_goose, prev_enc_state,
                               head_val=2 * (n_enemies + 1) + 4)

    # enemies
    for idx, goose in enumerate(prev_geese_info):
        prev_enc_state = _put_head(goose, prev_enc_state,
                                   head_val=2 * (n_enemies + 1) + 5 + idx)

    # centering
    prev_enc_state = _center(prev_enc_state, head_pos)

    # one hot encoding
    prev_enc_state = torch.nn.functional.one_hot(prev_enc_state)\
        .permute(2, 0, 1)

    return prev_enc_state


def _process_goose(goose, enc_state, head_val, body_val, tail_val):

    enc_state = _put_head(goose, enc_state, head_val)
    enc_state = _put_body(goose, enc_state, body_val)
    enc_state = _put_tail(goose, enc_state, tail_val)

    return enc_state


def _put_head(goose, enc_state, head_val):
    if len(goose) > 0:
        cell = goose[0]
        n_cols = enc_state.shape[1]
        enc_state[cell // n_cols, cell % n_cols] = head_val
    return enc_state


def _put_body(goose, enc_state, body_val):
    n_cols = enc_state.shape[1]
    for cell in goose[1:-1]:
        enc_state[cell // n_cols, cell % n_cols] = body_val
    return enc_state


def _put_tail(goose, enc_state, tail_val):
    if len(goose) > 1:
        cell = goose[-1]
        n_cols = enc_state.shape[1]
        enc_state[cell // n_cols, cell % n_cols] = tail_val
    return enc_state


def _center(curr_state, coord):

    y_pos = int(coord[0][0].detach().numpy())
    x_pos = int(coord[0][1].detach().numpy())
    n_rows, n_cols = curr_state.shape
    y_des = n_rows // 2
    x_des = n_cols // 2

    if y_des-y_pos <= 0:
        centered_curr_state = torch.cat((curr_state[y_pos - y_des:, :],
                                         curr_state[:y_pos - y_des, :]))
    else:
        centered_curr_state = torch.cat((curr_state[y_des + y_pos + 1:, :],
                                         curr_state[:y_des + y_pos + 1, :]))

    if x_des-x_pos <= 0:
        centered_curr_state = torch.cat((centered_curr_state[:, x_pos -
                                                             x_des:],
                                         centered_curr_state[:, :x_pos -
                                                             x_des]),
                                        dim=1)
    else:
        centered_curr_state = torch.cat((centered_curr_state[:, x_des +
                                                             x_pos + 1:],
                                         centered_curr_state[:, :x_des +
                                                             x_pos + 1]),
                                        dim=1)

    return centered_curr_state
