import json
import argparse
import os
from os.path import join

import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cfm.baselines.planet_models import bottle, VisualObservationModel, SSM
from cfm.dataset import PlanetDataset


def load_dataloaders():
    train_dset = PlanetDataset(join(args.root, 'train_data'), args.chunk_size)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_dset = PlanetDataset(join(args.root, 'test_data'), args.chunk_size)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, test_loader


def compute_loss(observations, actions, ssm, observation_model):
    # Input is B x T x ...., need T x B x ....
    observations = torch.transpose(observations, 0, 1).contiguous()
    actions = torch.transpose(actions, 0, 1).contiguous()

    # Create initial belief and state for time t = 0
    init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = ssm(
        init_state, actions[:-1], observations[1:])
    # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
    observation_loss = F.mse_loss(bottle(observation_model, (posterior_states,)), observations[1:],
                                  reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
    kl_loss = torch.max(
        kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2),
        free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
    if args.global_kl_beta != 0:
        kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs),
                                                       global_prior).sum(dim=2).mean(dim=(0, 1))
    # Calculate latent overshooting objective for t > 0
    if args.overshooting_kl_beta != 0:
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, args.chunk_size - 1):
            d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
            t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
            seq_pad = (0, 0, 0, 0, 0,
                       t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
            # Store (0) actions, (1) prior states, (2) posterior means, (3) posterior standard deviations and (4) sequence masks
            overshooting_vars.append((F.pad(actions[t:d], seq_pad), prior_states[t_],
                                      F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
                                      F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
                                      F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        overshooting_vars = tuple(zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        prior_states, prior_means, prior_std_devs = ssm(torch.cat(overshooting_vars[1], dim=0),
                                                        torch.cat(overshooting_vars[0], dim=1), None)
        seq_mask = torch.cat(overshooting_vars[4], dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(
            Normal(torch.cat(overshooting_vars[2], dim=1), torch.cat(overshooting_vars[3], dim=1)),
            Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
    return observation_loss + kl_loss, observation_loss, kl_loss


def train(ssm, observation_model, param_list, train_loader, optimizer, epoch):
    ssm.train()
    observation_model.train()

    train_losses, obs_losses, kl_losses = [], [], []
    pbar = tqdm(total=len(train_loader.dataset))
    for observations, actions in train_loader:
        observations, actions = observations.to(args.device), actions.to(args.device)
        loss, obs_loss, kl_loss = compute_loss(observations, actions, ssm, observation_model)

        if args.learning_rate_schedule != 0:
            for group in optimizer.param_groups:
                group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        optimizer.step()

        train_losses.append(loss.item())
        obs_losses.append(obs_loss.item())
        kl_losses.append(kl_loss.item())

        avg_loss = np.mean(train_losses[-50:])
        avg_obs_loss = np.mean(obs_losses[-50:])
        avg_kl_loss = np.mean(kl_losses[-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}, Obs Loss {avg_obs_loss:.4f}, KL Loss {avg_kl_loss:.4f}')
        pbar.update(observations.shape[0])
    pbar.close()


def test(ssm, observation_model, test_loader):
    ssm.eval()
    observation_model.eval()

    test_loss, test_obs, test_kl = 0, 0, 0
    for observations, actions in test_loader:
        observations, actions = observations.to(args.device), actions.to(args.device)
        with torch.no_grad():
            loss, obs_loss, kl_loss = compute_loss(observations, actions, ssm, observation_model)
            test_loss += loss * observations.shape[0]
            test_obs += obs_loss * observations.shape[0]
            test_kl += kl_loss * observations.shape[0]
    test_loss /= len(test_loader.dataset)
    test_obs /= len(test_loader.dataset)
    test_kl /= len(test_loader.dataset)

    print(f'Test Loss {test_loss:.4f}, Obs Loss {test_obs:.4f}, KL Loss {test_kl:.4f}')
    return test_loss.item()


def main():
    if 'rope' in args.root:
        action_dim = 4
    elif 'cloth' in args.root:
        action_dim = 5
    else:
        raise Exception('Invalid environment, or environment needed in root name')

    # Initialise model parameters randomly
    observation_model = VisualObservationModel(args.state_size, args.embedding_size, args.activation_function).to(
        device=args.device)
    ssm = SSM(args.state_size, action_dim, args.hidden_size, args.embedding_size, args.trans_type,
              activation_function=args.activation_function).to(args.device)
    param_list = list(ssm.parameters()) + list(observation_model.parameters())
    optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate,
                           eps=args.adam_epsilon)

    # Load data
    train_loader, test_loader = load_dataloaders()
    best_loss = float('inf')
    # Training (and testing)
    for epoch in range(args.epochs):
        train(ssm, observation_model, param_list, train_loader, optimizer, epoch)
        test_loss = test(ssm, observation_model, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            checkpoint = {
                'encoder': ssm.encoder,
                'trans': ssm.trans,
                'decoder': observation_model,
                'optimizer': optimizer,
            }
            torch.save(checkpoint, join(results_dir, 'checkpoint'))
            print('Saved with loss', best_loss)


if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser(description='PlaNet')
    parser.add_argument('--trans_type', type=str, default='mlp', help='linear | mlp | reparam_w')
    parser.add_argument('--root', type=str, default='data/pointmass')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--activation-function', type=str, default='relu',
                        help='Model activation function')
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E',
                        help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
    parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D',
                        help='Latent overshooting distance/latent overshooting weight for t = 1')
    parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1',
                        help='Latent overshooting KL weight for t > 1 (0 to disable)')
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                        help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
    parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value')
    parser.add_argument('--epochs', type=int, default=30, metavar='e', help='Number of epochs to train')
    # Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I',
                        help='Checkpoint interval (episodes)')
    parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
    args = parser.parse_args()
    args.overshooting_distance = min(args.chunk_size,
                                     args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
    #print(' ' * 26 + 'Options')
    #for k, v in vars(args).items():
    #    print(' ' * 26 + k + ': ' + str(v))

    # Setup
    results_dir = os.path.join('out', args.id)
    os.makedirs(results_dir, exist_ok=True)

    save_args = vars(args)
    save_args['script'] = 'train_planet'
    with open(join(results_dir, 'params.json'), 'w') as f:
        json.dump(save_args, f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = torch.device('cuda')
    global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device),
                          torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
    free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence

    main()
