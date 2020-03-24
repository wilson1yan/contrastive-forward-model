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


def train(encoder, trans, inverse_model, optimizer, train_loader, epoch, device):
    trans.train()

    stats = cu.Stats()
    pbar = tqdm(total=len(train_loader.dataset))
    parameters = list(encoder.parameters()) + list(trans.parameters())  + list(inverse_model.parameters())
    for batch in train_loader:
        obs, obs_pos, actions = [b.to(device) for b in batch]
        z, z_pos = encoder(obs), encoder(obs_pos)
        z_pred = trans(z, actions)
        action_pred = inverse_model(z, z_pos)

        trans_loss = F.mse_loss(z_pred, z_pos)
        inv_model_loss = F.mse_loss(action_pred, actions)
        loss = trans_loss + inv_model_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()

        stats.add('train_loss', loss.item())
        stats.add('trans_loss', trans_loss.item())
        stats.add('inv_model_loss', inv_model_loss.item())
        avg_loss = np.mean(stats['train_loss'][-50:])
        avg_trans_loss = np.mean(stats['trans_loss'][-50:])
        avg_inv_model_loss = np.mean(stats['inv_model_loss'][-50:])

        pbar.set_description(f'Epoch {epoch}, Train Loss {avg_loss:.4f}, '
                             f'Trans Loss {avg_trans_loss:.4f}, '
                             f'Inv Model Loss {avg_inv_model_loss:.4f}')
        pbar.update(obs.shape[0])
    pbar.close()


def test(encoder, trans, inverse_model, test_loader, epoch, device):
    trans.eval()

    test_loss, test_trans_loss, test_inv_model_loss = 0, 0, 0
    for batch in test_loader:
        with torch.no_grad():
            obs, obs_pos, actions = [b.to(device) for b in batch]
            z, z_pos = encoder(obs), encoder(obs_pos)
            z_pred = trans(z, actions)
            action_pred = inverse_model(z, z_pos)

            trans_loss = F.mse_loss(z_pred, z_pos)
            inv_model_loss = F.mse_loss(action_pred, actions)
            loss = trans_loss + inv_model_loss

            test_loss += loss * obs.shape[0]
            test_trans_loss += trans_loss * obs.shape[0]
            test_inv_model_loss += inv_model_loss * obs.shape[0]
    test_loss /= len(test_loader.dataset)
    test_trans_loss /= len(test_loader.dataset)
    test_inv_model_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}, Trans Loss: {test_trans_loss:.4f}, Inv Model Loss: {test_inv_model_loss:.4f}')
    return test_loss.item()


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    folder_name = join('out', args.name)
    if not exists(folder_name):
        os.makedirs(folder_name)

    save_args = vars(args)
    save_args['script'] = 'train_joint_dynamics'
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
    encoder = cm.Encoder(args.z_dim, obs_dim[0]).to(device)
    inverse_model = cm.InverseModel(args.z_dim, action_dim).to(device)
    trans = cm.Transition(args.z_dim, action_dim, trans_type=args.trans_type).to(device)
    parameters = list(encoder.parameters()) + \
                 list(trans.parameters()) + list(inverse_model.parameters())

    optimizer = optim.Adam(parameters, lr=args.lr)
    train_loader, test_loader = get_dataloaders()

    # Save training images
    batch = next(iter(train_loader))
    obs, obs_next, _ = batch
    imgs = torch.stack((obs, obs_next), dim=1).view(-1, *obs.shape[1:])
    cu.save_image(imgs * 0.5 + 0.5, join(folder_name, 'train_seq_img.png'), nrow=8)

    best_test_loss = float('inf')
    for epoch in range(args.epochs):
        train(encoder, trans, inverse_model, optimizer, train_loader, epoch, device)
        test_loss = test(encoder, trans, inverse_model, test_loader, epoch, device)

        if epoch % args.log_interval == 0:
            if test_loss <= best_test_loss:
                best_test_loss = test_loss

                checkpoint = {
                    'encoder': encoder,
                    'trans': trans,
                    'inverse_model': inverse_model,
                    'optimizer': optimizer,
                }
                torch.save(checkpoint, join(folder_name, 'checkpoint'))
                print('Saved models with loss', best_test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--root', type=str, default='data/pointmass', help='path to dataset (default: data/pointmass)')

    # Architecture Parameters
    parser.add_argument('--z_dim', type=int, default=8, help='latent dimension (default: 8)')
    parser.add_argument('--trans_type', type=str, default='linear',
                        help='linear | mlp | reparam_w | reparam_w_ortho_gs | reparam_w_ortho_cont | reparam_w_tanh (default: linear)')

    # Learning Parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='base learning rate for batch size 128 (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=50, help='default: 50')
    parser.add_argument('--log_interval', type=int, default=1, help='default: 1')
    parser.add_argument('--batch_size', type=int, default=128, help='default 128')

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='pixel_forward')
    args = parser.parse_args()

    main()
