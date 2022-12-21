import os
import ntpath
import time
from argparse import ArgumentParser

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
from utils.functions import train_xent, test_acc
from utils import loaders
from utils.misc import parse_range_tokens, dump_list_element_1line, get_num_parameters, pretty_seconds


#########################################
# arguments
#########################################
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', action='store_true', default=False)
parser.add_argument('--decay', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_steps', type=int, nargs='+', default=[100, 150])
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--test_epochs', type=str, default='', nargs='+',
                    help='epochs on which the model is tested. By default no epochs are chosen')

parser.add_argument('--model', type=str, choices=model_names, required=True)
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--save_model_path', type=str, default='')
parser.add_argument('--tag', type=str, default='', help='just a tag')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--test_datasets', type=str,  default='')

parser.add_argument('--basis_alpha', type=float, default=0.1)
parser.add_argument('--basis_scale', type=float, default=0.2)
parser.add_argument('--basis_radius', type=float, default=1.0)
parser.add_argument('--basis_num_displacements', type=int, default=4)
parser.add_argument('--basis_size', type=int, default=3)
parser.add_argument('--basis_mean', type=int, default=0)
args = parser.parse_args()

print("Args:")
for k, v in vars(args).items():
    print("  {}={}".format(k, v))

print(flush=True)

test_epochs = parse_range_tokens(args.test_epochs)
print('Test epochs: {}'.format(test_epochs or 'none'))
print(flush=True)


#########################################
# Data
#########################################
test_loader_pert = loaders.stl10_pert_test_loader(args.batch_size, args.data_dir)
test_loader = loaders.stl10_test_loader(args.batch_size, args.data_dir)


#########################################
# Model
#########################################
model = models.__dict__[args.model]
model = model(**vars(args))
##This code snippet is taken from SiamSE Sosnovik et.al
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys

    print('missing keys:{}'.format(missing_keys))
    print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    '''
    Old style model is stored with all names of parameters share common prefix 'module.'
    '''
    print('remove prefix "{}"'.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_pretrain(model, pretrained_path):
    print('load pretrained model from {}'.format(pretrained_path))

    
    pretrained_dict = torch.load(
        pretrained_path)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')


    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model



model = load_pretrain(model, args.save_model_path)
use_cuda = args.cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Device: {}'.format(device))

if use_cuda:
    num_gpus = torch.cuda.device_count()
    cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, range(num_gpus))
    print('model is using {} GPU(s)'.format(num_gpus))
print('num_params:', get_num_parameters(model))
print(flush=True)

start_time = time.time()



print('Testing...')
final_acc = test_acc(model, test_loader, device)
print('Acc@1: {:3.2f}%'.format(final_acc * 100))
#If testing on perturbed inputs
print('Testing Pert...')
final_acc_pert = test_acc(model, test_loader_pert, device)
print('Acc@1: {:3.2f}%'.format(final_acc_pert * 100))
end_time = time.time()
elapsed_time = end_time - start_time
time_per_epoch = elapsed_time / args.epochs




#########################################
# save results
#########################################


results = vars(args)
results.update({
    'dataset': 'stl10+',
    'acc': final_acc,
    'final_acc_pert': final_acc_pert
})

with open('test_results.yml', 'a') as f:
    f.write(dump_list_element_1line(results))
