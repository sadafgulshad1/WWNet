import torch
import torch.nn as nn
# from .attack_iterative import AttackIterative
# run_attack_iterative = AttackIterative()

def train_xent(model, optimizer, loader, device=torch.device('cuda')):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test_acc(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            # print ("batch: ",batch_idx)
            data, target = data.to(device), target.to(device)
            # adv_out = run_attack_iterative.run(data, target, model)

            # output = model(adv_out)
            output = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy
def test_acc_adv(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0

    for batch_idx, (data, target) in enumerate(loader):
        # print ("batch: ",batch_idx)
        data, target = data.to(device), target.to(device)
        adv_out = run_attack_iterative.run(data, target, model)

        output = model(adv_out)

        pred = output.argmax(1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy
