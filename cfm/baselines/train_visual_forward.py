import argparse
import json
import os
from os.path import join, exists

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from cfm.dataset import DynamicsDataset
import cfm.models as cm
import cfm.utils as cu


def get_dataloaders():
    train_dset = DynamicsDataset(root=join(args.root, 'train_data'))
    train_loader = data.DataLoader(train_dset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4,
                                   pin_memory=True,)

    test_dset = DynamicsDataset(root=join(args.root, 'test_data'))
    test_loader = data.DataLoader(test_dset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4,
                                  pin_memory=True)


    return train_loader, test_loader


def train(trans, optimizer, train_loader, epoch, device):
    trans.train()

    stats = cu.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]
        loss = F.mse_loss(trans(obs, actions), obs_pos)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trans.parameters(), 20)
        optimizer.step()

        stats.add('train_loss', loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()


def test(trans, test_loader, epoch, device):
    trans.eval()

    test_loss = 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            loss = F.mse_loss(trans(obs, actions), obs_pos)
            test_loss += loss * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}')
    return test_loss.item()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    save_args = vars(args)
    save_args['script'] = 'train_visual_forward'
    with open(join(folder_name, 'params.json'), 'w') as f:
        json.dump(save_args, f)

    obs_dim = (3, 64, 64)
    if 'rope' in args.root:
        action_dim = 4
    elif 'cloth' in args.root:
        action_dim = 5
    else:
        raise Exception('Invalid environment, or environment needed in root name')

    device = torch.device('cuda')
    encoder = nn.Identity() # Just to keep the interface consistent for eval
    trans = cm.PixelForwardModel(obs_dim, action_dim, learn_delta=args.learn_delta).to(device)
    parameters = list(trans.parameters())

    optimizer = optim.Adam(parameters, lr=args.lr)
    train_loader, test_loader = get_dataloaders()

    # Save training images
    batch = next(iter(train_loader))
    obs, obs_next, _ = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    cu.save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        train(trans, optimizer, train_loader, epoch, device)
        test_loss = test(trans, test_loader, epoch, device)

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                checkpoint = {
                    'encoder': encoder,
                    'trans': trans,
                    'optimizer': optimizer,
                }
                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/pointmass', help='path to dataset (default: data/pointmass)')

    # Architecture Parameters
    parser.add_argument('--learn_delta', action='store_true', help='learn image delta instead of next image')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=50, help='default: 50')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=128, help='default 128')

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='visual_forward')
    args = parser.parse_args()

    main()
