import torch
from torch.distributions.normal import Normal


def neg_log_normal_pdf(x, mu, sigma, reduce_metric='mean'):
    # reshape mu and sigma to match the shape of x if only one mu and sigma is given
    if mu.ndim == 1:
        mu = mu.repeat(x.shape[0], 1).reshape(x.shape)
        sigma = sigma.repeat(x.shape[0], 1).reshape(x.shape)

    distribution = Normal(mu, sigma)
    neg_log_prob = torch.neg(distribution.log_prob(x))
    if reduce_metric == 'mean':
        neg_log_prob = torch.mean(neg_log_prob)

    return neg_log_prob

def cross_entropy_likelihood(x, y, net, loss_func):
    y_hat = net(x)
    return loss_func(y_hat, y)
