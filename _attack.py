import torch
import torch.nn.functional as F

def fgsm(model, data, epsilon, device):
    model = model.to(device)
    model.eval()
    ori_x, adv_x = [], []

    for x in data:
        x = x.to(device)
        x.requires_grad = True

        output = model(x)
        _, y = output.topk(1)
        loss = F.nll_loss(output, torch.tensor([y]).to(device))
        model.zero_grad()
        loss.backward()
        perturbed_x = x + epsilon * x.grad.data.sign()

        ori_x.append(x.squeeze(0).detach().cpu().numpy())
        adv_x.append(perturbed_x.squeeze(0).detach().cpu().numpy())

    return ori_x, adv_x
