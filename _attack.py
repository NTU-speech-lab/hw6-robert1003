import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd.gradcheck import zero_gradients

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
        
        '''
        if y != py:
            wrong += 1
            
            ori_x.append(x.squeeze(0).detach().cpu())
            adv_x.append(x.squeeze(0).detach().cpu())
            
            continue
        '''

        loss = F.cross_entropy(output, torch.tensor([y]).to(device))
        loss.backward()
        perturbed_x = x + epsilon * x.grad.data.sign()

        output = model(perturbed_x)
        _, py = output.topk(1)
        
        if py != y:
            succ += 1
        else:
            fail += 1
        
        ori_x.append(x.squeeze(0).detach().cpu())
        adv_x.append(perturbed_x.squeeze(0).detach().cpu())

    print('wrong: {}, succ: {}, fail: {} -> final acc: {}, succ attack rate: {}'.format(wrong, succ, fail, fail / (wrong + succ + fail), (wrong + succ) / (wrong + succ + fail)))
        
    return ori_x, adv_x

def test(model, x, y, transform, inv_transform, device):
    model.eval()
    x = transform(inv_transform(x)).unsqueeze(0).to(device)
    _, py = model(x).topk(1)
    return (y != py).all()

def get_pred(model, img, transform, device):
    model.eval()
    _, py = model(transform(img).unsqueeze(0).to(device)).topk(1)
    return py

def deepfool(model, dataLoader, transform, inv_transform, start, end, eps, max_iter, overshoot, num_classes, device):
    model = model.to(device)
    model.eval()
    ori_x, adv_x = [], []

    wrong, succ, fail, linf = 0, 0, 0, 0
    dataIter = tqdm(dataLoader, desc='[*] Attacking')
    for idx, (img, y) in enumerate(dataIter):
        if idx < start or idx > end:
            continue

        img, y = img.to(device), y.to(device)
        upper = img + eps
        lower = img - eps

        img.requires_grad = True
        labels = model(img).detach().cpu().numpy().flatten().argsort()[::-1][0:num_classes]
        ori_label = labels[0]

        if ori_label != y:
            wrong += 1
            ori_x.append(None)
            adv_x.append(None)
            '''
            ori_x.append(inv_transform(img.cpu().squeeze(0)))
            adv_x.append(inv_transform(img.cpu().squeeze(0)))
            if y == get_pred(model, adv_x[-1], transform, device):
                print('w', idx)
            linf += np.linalg.norm((np.array(ori_x[-1]).astype('int64') - np.array(adv_x[-1]).astype('int64')).flatten(), np.inf)
            '''
            continue

        x = img.detach()
        x.requires_grad = True
        x_preds = model(x)
        k_i = ori_label

        for _ in range(max_iter):
            if k_i != ori_label:
                if test(model, x.cpu()[0], y, transform, inv_transform, device):
                    break
                else:
                    r2 = r / np.max(r) * eps * 0.2
                    '''
                    eps -= 1e-7
                    upper = img.detach() + eps
                    lower = img.detach() - eps
                    '''
                    x = x.detach() + torch.from_numpy(r2).to(device)
                    x = torch.min(torch.max(x, lower), upper)
                    x.requires_grad = True
                    x_preds = model(x)
                    k_i = np.argmax(x_preds.detach().cpu().numpy().flatten())

            pert = np.inf
            x_preds[0, ori_label].backward(retain_graph=True)
            ori_grad = x.grad.detach().cpu().numpy()

            for k in range(1, num_classes):
                zero_gradients(x)
                x_preds[0, labels[k]].backward(retain_graph=True)
                cur_grad = x.grad.detach().cpu().numpy()

                w_k = cur_grad - ori_grad
                f_k = (x_preds[0, labels[k]] - x_preds[0, ori_label]).detach().cpu().numpy()
                pert_k = np.abs(f_k) / (np.linalg.norm(w_k.flatten(), 1) + 1e-8)

                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r = (pert + 1e-4) * (np.sign(w) * 0.5)#w / np.linalg.norm(w)
            x = x.detach() + (1 + overshoot) * torch.from_numpy(r).to(device)
            x = torch.min(torch.max(x, lower), upper)

            x.requires_grad = True
            x_preds = model(x)
            k_i = np.argmax(x_preds.detach().cpu().numpy().flatten())

        ori_x.append(inv_transform(img.cpu().squeeze(0)))
        adv_x.append(inv_transform(x.cpu()[0]))
        if y == get_pred(model, adv_x[-1], transform, device):
            print(idx)
            fail += 1
        else:
            succ += 1

        linf += np.linalg.norm((np.array(ori_x[-1]).astype('int64') - np.array(adv_x[-1]).astype('int64')).flatten(), np.inf)

    print('wrong: {}, succ: {}, fail: {} -> final acc: {}, succ attack rate: {}'.format(wrong, succ, fail, fail / (wrong + succ + fail), (wrong + succ) / (wrong +     succ + fail)))
    print('linf: {}'.format(linf / (wrong + succ + fail)))
    return ori_x, adv_x
