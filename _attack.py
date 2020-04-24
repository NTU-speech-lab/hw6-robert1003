import torch
import torch.nn.functional as F
from tqdm import tqdm

def fgsm(model, dataLoader, epsilon, device):
    model = model.to(device)
    model.eval()
    ori_x, adv_x = [], []

    wrong, succ, fail = 0, 0, 0
    dataIter = tqdm(dataLoader, desc='[*] Attack')
    for (x, y) in dataIter:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True

        output = model(x)
        _, py = output.topk(1)
        
        if y != py:
            wrong += 1
            
            ori_x.append(x.squeeze(0).detach().cpu().numpy())
            adv_x.append(x.squeeze(0).detach().cpu().numpy())
            
            continue
        
        loss = F.cross_entropy(output, torch.tensor([y]).to(device))
        loss.backward()
        perturbed_x = x + epsilon * x.grad.data.sign()

        output = model(perturbed_x)
        _, py = output.topk(1)
        
        if py != y:
            succ += 1
        else:
            fail += 1
        
        ori_x.append(x.squeeze(0).detach().cpu().numpy())
        adv_x.append(perturbed_x.squeeze(0).detach().cpu().numpy())

    print('wrong: {}, succ: {}, fail: {} -> final acc: {}, succ attack rate: {}'.format(wrong, succ, fail, fail / (wrong + succ + fail), (wrong + succ) / (wrong + succ + fail)))
        
    return ori_x, adv_x
