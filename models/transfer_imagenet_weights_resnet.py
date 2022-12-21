import torch
import torch.nn.functional as F
import numpy as np


def tikhonov_reg_lstsq(A, B, eps=1e-12):
    '''|A x - B| + |Gx| -> min_x
    '''
    W = A.shape[1]
    A_inv = np.linalg.inv(A.T @ A + eps * np.eye(W)) @ A.T
    return A_inv @ B


def copy_state_dict_bn(dict_target, dict_origin, key_target, key_origin):
    for postfix in ['weight', 'bias', 'running_mean', 'running_var']:
        dict_target[key_target + '.' + postfix] = dict_origin[key_origin + '.' + postfix]


def copy_state_dict_conv_hh_1x1(dict_target, dict_origin, key_target, key_origin):
    dict_target[key_target + '.weight'] = dict_origin[key_origin + '.weight']
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_fc(dict_target, dict_origin, key_target, key_origin):
    dict_target[key_target + '.weight'] = dict_origin[key_origin + '.weight']
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh_1x1_interscale(dict_target, dict_origin, key_target, key_origin):
    w_original = dict_target[key_target + '.weight']
    w_original *= 0
    w_original[:, :, 0] = dict_origin[key_origin + '.weight']
    dict_target[key_target + '.weight'] = w_original
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_zh(dict_target, dict_origin, key_target, key_origin, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    try:

        basis = dict_target[key_target + '.basis.basis'][:, 0]

        dict_target[key_target + '.weight'] = _approximate_weight(basis, weight, eps)
    except KeyError:
        dict_target[key_target + '.weight'] = weight

    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def copy_state_dict_conv_hh(dict_target, dict_origin, key_target, key_origin, eps=1e-12):
    weight = dict_origin[key_origin + '.weight']
    try:
        basis = dict_target[key_target + '.basis.basis'][:, 0]
        x = torch.zeros_like(dict_target[key_target + '.weight'])
        x[:, :, 0] = _approximate_weight(basis, weight, eps)
    except KeyError:
        x = weight

    dict_target[key_target + '.weight'] = x
    if key_target + '.bias' in dict_target:
        assert key_origin + '.bias' in dict_origin
        dict_target[key_target + '.bias'] = dict_origin[key_origin + '.bias']


def _approximate_weight(basis, target_weight, eps=1e-12):
    C_out, C_in, h, w = target_weight.shape

    B, H, W = basis.shape
    with torch.no_grad():
        basis = F.pad(basis, [(w - W) // 2, (w - W) // 2, (h - W) // 2, (h - H) // 2])
        target_weight = target_weight.view(C_out * C_in, h * w).detach().cpu().numpy()
        basis = basis.reshape(B, h * w).detach().cpu().numpy()
    # print('>>>>>', basis.shape)
    assert basis.shape[0] == basis.shape[1]

    matrix_rank = np.linalg.matrix_rank(basis)

    if matrix_rank == basis.shape[0]:
        x = np.linalg.solve(basis.T, target_weight.T).T
    else:
        print('  !!! basis is incomplete. rank={} < num_funcs={}. '
              'weights are approximateb by using '
              'tikhonov regularization'.format(matrix_rank, basis.shape[0]))
        x = tikhonov_reg_lstsq(basis.T, target_weight.T, eps=eps).T

    diff = np.linalg.norm(x @ basis - target_weight)
    norm = np.linalg.norm(target_weight) + 1e-12
    print('  relative_diff={:.1e}, abs_diff={:.1e}'.format(diff / norm, diff))
    x = torch.Tensor(x)
    x = x.view(C_out, C_in, B)
    return x


def convert_param_name_to_layer(name):
    layer_name = '.'.join(name.split('.')[:-1])

    if 'basis' in name:
        return layer_name, 'save'
    if 'bn' in name:
        return layer_name, 'bn'
    if 'conv1' in name:   ##Uncomment this for a standard Resnet18
         return layer_name, 'save'
    if 'conv' in name:
        return layer_name, 'conv'
    if 'downsample' in name:
        return layer_name, 'conv'
    if 'fc' in name:
        return layer_name, 'save'
    if 'shortcut' in name:
        return layer_name, 'save'
    if 'linear' in name:
        return layer_name, 'save'

    print(name)
    raise NotImplementedError


def transfer_weights(src_state_dict, dest_state_dict):
    print('Transferring weights....')
    #     print ("src",(src_state_dict.keys()))
    keys = list(dest_state_dict.keys())
#     print ("dest",(keys))
    layers = list(set(['.'.join(key.split('.')[:-1]) for key in keys]))
    layers_repr = [convert_param_name_to_layer(name) for name in keys]
    layers_repr = list(set(layers_repr))


    for layer_name, layer_type in layers_repr:
        # print('Layer {}:'.format(layer_name))

        if layer_type == 'bn':
            copy_state_dict_bn(dest_state_dict, src_state_dict, layer_name, layer_name)

        if layer_type == 'fc':
            copy_state_dict_fc(dest_state_dict, src_state_dict, layer_name, layer_name)

        if layer_type == 'conv':
            weight = dest_state_dict[layer_name + '.weight']
            weight_src = src_state_dict[layer_name + '.weight']
            if len(weight.shape) == 1:
                copy_state_dict_bn(dest_state_dict, src_state_dict, layer_name, layer_name)
                continue


            if weight.shape[-1] == weight.shape[-2] == 1:
                if len(weight.shape) == 4:
                    copy_state_dict_conv_hh_1x1(
                        dest_state_dict, src_state_dict, layer_name, layer_name)
                elif len(weight.shape) == 5:
                    copy_state_dict_conv_hh_1x1_interscale(
                        dest_state_dict, src_state_dict, layer_name, layer_name)
                else:
                    raise NotImplementedError
            elif len(weight.shape) == 4:
                copy_state_dict_conv_hh(dest_state_dict, src_state_dict, layer_name, layer_name)
            else:
                copy_state_dict_conv_zh(dest_state_dict, src_state_dict, layer_name, layer_name)
        if layer_type == 'save':
            pass

    return dest_state_dict
