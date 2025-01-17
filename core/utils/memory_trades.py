import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from core.metrics import accuracy


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def cos_similarity_mean(x, x_adv, x_prime, mask=None):
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    sigma = x_adv - x
    sigma_prime = x_prime - x

    sigma = sigma.flatten(start_dim =1)
    sigma_prime = sigma_prime.flatten(start_dim =1)

    sim = cos_sim(sigma, sigma_prime)

    if mask is not None:
        sim = sim * mask

    return sim.mean().item()



def memory_trades_loss(model, x_natural, y, x_prime, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, beta_prime=1.0, 
                attack='linf-pgd', attack_loss='kl', weighted=False, sim=False, ema_xprime = False):
    """
    TRADES training (Zhang et al, 2019).
    """
  
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    p_natural = F.softmax(model(x_natural), dim=1)
    p_x_prime = F.softmax(model(x_prime), dim=1)
    
    if attack == 'linf-pgd':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if attack_loss == 'memory-kl':
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural) +\
                          (beta_prime/beta) * criterion_kl(F.log_softmax(model(x_adv), dim=1), p_x_prime)
                elif attack_loss == 'kl':
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
                
                elif attack_loss == 'ce':
                    loss_kl = F.cross_entropy(model(x_adv), y)
                else:
                    raise ValueError(f'Attack loss={attack_loss} not supported')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    elif attack == 'l2-pgd':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward(retain_graph=True)
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f'Attack={attack} not supported for TRADES training!')
    
    if sim:
        cos_sim = cos_similarity_mean(x_natural, x_adv, x_prime)
    
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)
    logits_x_prime = model(x_prime)
    loss_natural = F.cross_entropy(logits_natural, y)
    log_softmax_adv_logits = F.log_softmax(logits_adv, dim=1)
    # loss_robust = (1.0 / batch_size) * criterion_kl(log_softmax_adv_logits,
    #                                                 F.softmax(logits_natural, dim=1))
    
    if weighted:
        kl_without_reduction = nn.KLDivLoss(reduction='none')
        x_prime_probs = F.softmax(logits_x_prime, dim=1)
        x_probs = F.softmax(logits_natural, dim=1)
        # x_prime_false_preds = (x_prime_probs.argmax(dim=1) != y).float()
        x_prime_true_probs = torch.gather(x_prime_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        x_prime_max_probs = x_prime_probs.max(dim=1)[0]
        x_true_preds = (x_probs.argmax(dim=1) == y).float()
        pos_weights = x_true_preds * (1 + x_prime_max_probs - x_prime_true_probs)
        # pos_weights = beta_prime * pos_weights

        ALPHA = 0.4
        x_false_preds = 1 - x_true_preds
        x_true_probs = torch.gather(x_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        x_max_probs =  x_probs.max(dim=1)[0]
        neg_mask = (x_max_probs - x_true_probs) <= ALPHA
        neg_weights = x_false_preds * neg_mask * (1 + x_true_probs - x_max_probs)

        weights = pos_weights + neg_weights
        
        #XPRIME WEIGHTED
        # memory_loss = (1.0 / batch_size) * torch.sum(torch.sum(kl_without_reduction\
        #           (F.log_softmax(logits_x_prime, dim=1), F.softmax(logits_natural, dim=1)),
        #             dim=1) * (1.0000001 - x_prime_true_probs))
        
        loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(kl_without_reduction\
                  (log_softmax_adv_logits, F.softmax(logits_natural, dim=1)),
                    dim=1) * (0.0000001 + weights))

        # loss_robust = (1.0 / batch_size) * criterion_kl(log_softmax_adv_logits,
        #                                                 F.softmax(logits_natural, dim=1))
        
        # memory_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_x_prime, dim=1),
        #                                             F.softmax(logits_natural, dim=1))
        
        loss = loss_natural + beta * loss_robust
        
    else:
        memory_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_x_prime, dim=1),
                                                        F.softmax(logits_natural, dim=1))
        loss_robust = (1.0 / batch_size) * criterion_kl(log_softmax_adv_logits,
                                                        F.softmax(logits_natural, dim=1))
        
        loss = loss_natural + beta * loss_robust + beta_prime * memory_loss
    
    
    batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(y, logits_natural.detach()), 
                     'adversarial_acc': accuracy(y, logits_adv.detach()), 'memory_loss':0.0}
    if sim:
        batch_metrics.update({'cos_sim':cos_sim})
    
    if ema_xprime:
        ema_coef = 0.8 # avg on 1 /(1 - ema_coef) prev attacks
        x_adv = (ema_coef * x_prime) + ((1 - ema_coef) * x_adv)
        
    return loss, batch_metrics, x_adv.detach().cpu()
