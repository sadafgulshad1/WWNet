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
parser.add_argument('--basis_type', type=str, default='hermite_circular')
parser.add_argument('--test_datasets', type=str, nargs='+', required=False, default=[])

parser.add_argument('--basis_alpha', type=float, default=0.1)
parser.add_argument('--basis_radius', type=float, default=1.0)
parser.add_argument('--basis_scale', type=float, default=0.2)
parser.add_argument('--basis_num_displacements', type=int, default=4)
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
train_loader = loaders.stl10_train_loader(args.batch_size, args.data_dir)
test_loader = loaders.stl10_test_loader(args.batch_size, args.data_dir)

print('Train:')
print(train_loader.dataset)
print()
print('Test:')
print(test_loader.dataset)
print(flush=True)

#########################################
# Model
#########################################
model = models.__dict__[args.model]
model = model(**vars(args))
print(model)

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


#########################################
# optimizer
#########################################
parameters = filter(lambda x: x.requires_grad, model.parameters())
if args.optim == 'adam':
    optimizer = optim.Adam(parameters, lr=args.lr)
if args.optim == 'sgd':
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay, nesterov=args.nesterov)
    
lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.008,max_lr=0.12)

#########################################
# training
#########################################
print('\nTraining\n' + '-' * 30)

start_time = time.time()
best_acc = 0.0

for epoch in range(args.epochs):
    train_xent(model, optimizer, train_loader, device)
    if epoch % 50 == 0:
        current_time = time.time() - start_time
        eta = current_time * (args.epochs - epoch - 1) / (epoch + 1)
        print('Epoch {:3d}/{:3d}| Time={:.0f}s| '
              'ETA={}'.format(epoch + 1, args.epochs, current_time, pretty_seconds(eta)), flush=True)

    if epoch % 30 ==0:
        acc = test_acc(model, test_loader, device)
        print('Epoch {:3d}/{:3d}| '
              'Acc@1: {:3.2f}%'.format(epoch + 1, args.epochs, 100 * acc), flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),args.save_model_path+'.pth')
            torch.save(model,args.save_model_path)
        lr_scheduler.step()

print('-' * 30)
print('Training is finished')
print('Testing...')
final_acc = test_acc(model, test_loader, device)
print('Final Acc@1: {:3.2f}%'.format(final_acc * 100))
print('Best Acc@1: {:3.2f}%'.format(best_acc * 100), flush=True)
end_time = time.time()
elapsed_time = end_time - start_time
time_per_epoch = elapsed_time / args.epochs

results = vars(args)
results.update({
    'dataset': 'stl10+',
    'elapsed_time': int(elapsed_time),
    'time_per_epoch': int(time_per_epoch),
    'num_parameters': int(get_num_parameters(model)),
    'acc': best_acc
})

with open('results.yml', 'a') as f:
    f.write(dump_list_element_1line(results))
