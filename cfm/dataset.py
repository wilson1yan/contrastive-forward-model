import torch.utils.data as data
import h5py
import torch
import numpy as np
import pickle as pkl
import os
import glob
from os.path import join, dirname

from torchvision.datasets.folder import default_loader


class PlanetDataset(data.Dataset):
    def __init__(self, root, chunk_size, transform=None):
        self.root = root
        with open(join(root, 'pos_neg_pairs_1.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.image_paths = data['all_images']

        self.chunk_size = chunk_size
        self.trajectories = glob.glob(join(root, 'run*'))
        f = h5py.File(join(root, 'images.hdf5'), 'r')
        self.images = f['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        self.transform = transform

        if 'rope' in root:
            self.mean = np.array([0.5, 0.5, 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1])
        elif 'cloth' in root:
            self.mean = np.array([0.5, 0.5, 0., 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1, 1])
        else:
            raise Exception('Invalid environment, or environment needed in root name')

        self.n_trajectories = len(self.trajectories)
        self.traj_lengths = [len(glob.glob(join(r, 'img_*.png'))) for r in self.trajectories]
        assert all([self.chunk_size <= traj_length for traj_length in self.traj_lengths])
        self.n_per_traj = [traj_length - self.chunk_size
                           for traj_length in self.traj_lengths]

    def _get_image(self, path):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return sum(self.n_per_traj)

    def __getitem__(self, index):
        traj = 0
        while index >= self.n_per_traj[traj]:
            index -= self.n_per_traj[traj]
            traj += 1
        offset = index
        run_folder = self.trajectories[traj]
        actions = np.load(join(run_folder, 'actions.npy'))[offset:offset + self.chunk_size, 0]
        actions = (actions - self.mean[None, :]) / self.std[None, :]

        observations = []
        for t in range(self.chunk_size):
            img_path = join(run_folder, f'img_{str(offset + t).zfill(4)}_000.png')
            observations.append(self._get_image(img_path))
        observations = np.stack(observations, axis=0)

        return torch.FloatTensor(observations), torch.FloatTensor(actions)


class DynamicsDataset(data.Dataset):
    """
    Dataset that returns current state / next state image pairs
    """

    def __init__(self, root, s=1):
        self.root = root
        self.s = s # Same 's' as in process_dataset.py (number of timesteps between current / next pairs)

        with open(join(root, f'pos_neg_pairs_{s}.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.pos_pairs = data['pos_pairs']
        self.image_paths = data['all_images']

        # Load entire dataset into memory as opposed to accessing disk with hdf5
        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        # Normalization of actions for different datasets
        if 'rope' in root:
            self.mean = np.array([0.5, 0.5, 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1])
        elif 'cloth' in root:
            self.mean = np.array([0.5, 0.5, 0., 0., 0.])
            self.std = np.array([0.5, 0.5, 1, 1, 1])
        else:
            raise Exception('Invalid environment, or environment needed in root name')

    def _get_image(self, path):
        img = self.images[self.img2idx[path]]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, index):
        obs_file, obs_next_file, action_file = self.pos_pairs[index]
        obs, obs_next = self._get_image(obs_file), self._get_image(obs_next_file)
        actions = np.load(action_file)

        fsplit = obs_next_file.split('_')
        t = int(fsplit[-2])
        k = int(fsplit[-1].split('.')[0])
     #   assert k == 0
     #   assert t - self.s >= 0

     #   action = actions[t - self.s:t, k]
        assert self.s == 1
        action = actions[t - 1, k]
        action = (action - self.mean) / self.std

        return obs, obs_next, torch.FloatTensor(action)


class ImageDataset(data.Dataset):
    """
    Standard image dataset with added functionality to include the
    original MuJoCo state of the observation
    """

    def __init__(self, root, include_state=False, loader=default_loader,
                  transform=None):
        self.root = root
        self.include_state = include_state
        self.loader = loader
        self.transform = transform

        with open(join(root, 'pos_neg_pairs_1.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.image_paths = data['all_images']

        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images'][:]
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

    def _get_image(self, index):
        img = self.images[index]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.include_state:
            img_path = self.image_paths[index]
            folder = dirname(img_path)
            states = np.load(join(folder, 'env_states.npy'))
            t = int(img_path.split('_')[-2])
            return self._get_image(index), self.transform(self.loader(self.image_paths[index])), torch.FloatTensor(states[t])
        return self._get_image(index)

    def get_item_by_path(self, path):
        return self[self.img2idx[path]]


class TrajectoryDataset(data.Dataset):
    """
    Dataset that returns full trajectories with observations, actions, and
    true simulation states
    """
    def __init__(self, root):
        super().__init__()
        self.root = root
        with open(join(root, 'pos_neg_pairs_1.pkl'), 'rb') as f:
            data = pkl.load(f)
        self.image_paths = data['all_images']
        self.images = h5py.File(join(root, 'images.hdf5'), 'r')['images']
        self.img2idx = {self.image_paths[i]: i for i in range(len(self.image_paths))}

        self.runs = list(set([path.split('/')[-2] for path in self.image_paths]))

    def _get_image(self, index):
        img = self.images[index]
        img = img.astype('float32') / 255
        img = (img - 0.5) / 0.5
        return torch.FloatTensor(img)

    def _get_image_by_path(self, path):
        return self._get_image(self.img2idx[path])

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, index):
        run_id = self.runs[index]
        folder = join(self.root, run_id)

        img_files = glob.glob(join(self.root, run_id, 'img_*_000.png'))
        img_files = sorted(img_files)
        assert len(img_files) > 0

        obs = torch.stack([self._get_image_by_path(f) for f in img_files], dim=0)
        actions = np.load(join(folder, 'actions.npy'))[:, 0]
        states = np.load(join(folder, 'env_states.npy'))
        return obs, torch.FloatTensor(actions), torch.FloatTensor(states)
