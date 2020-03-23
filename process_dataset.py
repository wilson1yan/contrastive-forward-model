import sys
import os
from os.path import join, dirname, basename
import glob
import shutil
import pickle
import itertools

import numpy as np
from tqdm import tqdm
import h5py

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader as loader


s = 1 # number of timesteps between current image / next image pairs

def partition_dataset(root):
    """
    Partitions a folder of runs into train and test set folders
    """
    os.makedirs(join(root, 'train_data'))
    os.makedirs(join(root, 'test_data'))

    runs = glob.glob(join(root, 'run*'))
    threshold = int(len(runs) * 0.8)

    train_runs = runs[:threshold]
    test_runs = runs[threshold:]

    for name, subset_runs in zip(('train_data', 'test_data'), (train_runs, test_runs)):
        for run in subset_runs:
            current_path = run
            dest_path = join(dirname(run), name, basename(run))
            shutil.move(current_path, dest_path)


def compute_image_pairs(root):
    # Precomputes positive image pairs, and hard negatives if present
    runs = glob.glob(join(root, 'run*'))
    runs = sorted(runs)

    neg_samples_same_t = dict()
    neg_samples_same_traj = dict()
    pos_pairs = []
    all_images = []
    for run in tqdm(runs):
        action_file = join(run, 'actions.npy')
        np.load(action_file) # just to make sure the path is right

        neg_samples_same_t[run] = dict()
        neg_samples_same_traj[run] = dict()

        images = glob.glob(join(run, '*.png'))
        images = sorted(images)
        all_images.extend(images)
        images = org_images(images)
        for t in itertools.count():
            if t + s not in images:
                break
            for k in images[t+s]:
                pos_pairs.append((images[t][0], images[t+s][k], action_file))

            neg_samples_same_t[run][t] = [images[t+s][k] for k in images[t+s]]
            neg_samples_same_traj[run][t] = []
            for t_tmp in images:
                if t_tmp == t or t_tmp == t+s:
                    continue
                for k_tmp in images[t_tmp]:
                    neg_samples_same_traj[run][t].append(images[t_tmp][k_tmp])

    data = dict(pos_pairs=pos_pairs, neg_samples_t=neg_samples_same_t,
                neg_samples_traj=neg_samples_same_traj, all_images=all_images)
    with open(join(root, f'pos_neg_pairs_{s}.pkl'), 'wb') as f:
        pickle.dump(data, f)


def org_images(images):
    t_k = dict()
    for image in images:
        img_split = image.split('_')
        t = int(img_split[-2])
        k = int(img_split[-1].split('.')[0])
        if t not in t_k:
            t_k[t] = dict()
        assert k not in t_k[t]
        t_k[t][k] = image
    return t_k


def load_images(root):
    # Preloads images into an hdf5 dataset for faster access and training
    # Images are already normalized
    with open(join(root, f'pos_neg_pairs_{s}.pkl'), 'rb') as f:
        data = pickle.load(f)
    all_images = data['all_images']

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dset = h5py.File(join(root, 'images.hdf5'), 'x')
    dset.create_dataset('images', (len(all_images), 3, 64, 64), 'uint8')
    stored = []
    for i, img in enumerate(tqdm(all_images)):
        img = transform(loader(img))
        img = img.numpy() * 0.5 + 0.5
        img *= 255
        img = img.astype(np.uint8)
        dset['images'][i] = img


if __name__ == '__main__':
    roots = sys.argv[1:]
    for root in roots:
        partition_dataset(root)

        compute_image_pairs(join(root, 'train_data'))
        compute_image_pairs(join(root, 'test_data'))

        load_images(join(root, 'train_data'))
        load_images(join(root, 'test_data'))
