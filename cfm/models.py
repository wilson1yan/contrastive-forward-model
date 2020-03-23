import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, z_dim, channel_dim):
        super().__init__()

        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class Transition(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim, trans_type='linear'):
        super().__init__()
        if trans_type in ['linear', 'mlp']:
            self.model = TransitionSimple(z_dim, action_dim, trans_type=trans_type)
        elif 'reparam_w' in trans_type:
            self.model = TransitionParam(z_dim, action_dim, hidden_sizes=[64, 64],
                                         orthogonalize_mode=trans_type)
        else:
            raise Exception('Invalid trans_type:', trans_type)

    def forward(self, z, a):
        return self.model(z, a)


class TransitionSimple(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, trans_type='linear'):
        super().__init__()
        self.trans_type = trans_type
        self.z_dim = z_dim

        if self.trans_type == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 64
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, z_dim)
            )
        else:
            raise Exception('Invalid trans_type', trans_type)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x


class TransitionParam(nn.Module):
    prefix = 'transition'

    def __init__(self, z_dim, action_dim=0, hidden_sizes=[], orthogonalize_mode='reparam_w'):
        super().__init__()
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.orthogonalize_mode = orthogonalize_mode

        if orthogonalize_mode == 'reparam_w_ortho_cont':
            self.model = MLP(z_dim + action_dim, z_dim * (z_dim - 1), hidden_sizes=hidden_sizes)
        else:
            self.model = MLP(z_dim + action_dim, z_dim * z_dim, hidden_sizes=hidden_sizes)

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        if self.orthogonalize_mode == 'reparam_w':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
        elif self.orthogonalize_mode == 'reparam_w_ortho_gs':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim)  # b x z_dim x z_dim
            Ws = orthogonalize_gs(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_ortho_cont':
            Ws = self.model(x).view(x.shape[0], self.z_dim, self.z_dim - 1)  # b x z_dim x z_dim - 1
            Ws = orthogonalize_cont(Ws, self.z_dim)
        elif self.orthogonalize_mode == 'reparam_w_tanh':
            Ws = torch.tanh(self.model(x)).view(x.shape[0], self.z_dim, self.z_dim) / math.sqrt(self.z_dim)
        else:
            raise Exception('Invalid orthogonalize_mode:', self.orthogonalize_mode)
        return torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1) # b x z_dim


# Gram-Schmidt
def orthogonalize_gs(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)
    return Ws_new


def orthogonalize_cont(Ws, z_dim):
    Ws_new = Ws[:, :, [0]] / torch.norm(Ws[:, :, [0]], dim=1, keepdim=True)  # b x z_dim x 1
    for k in range(1, z_dim - 1):
        v, us = Ws[:, :, [k]], Ws_new.permute(0, 2, 1)  # b x z_dim x 1, b x k x z_dim
        dot = torch.bmm(us, v)  # b x k x 1
        diff = (us * dot).sum(dim=1)  # b x z_dim
        u = Ws[:, :, k] - diff  # b x z_dim
        u = u / torch.norm(u, dim=1, keepdim=True)
        Ws_new = torch.cat((Ws_new, u.unsqueeze(-1)), dim=-1)

    # Ws_new is b x z_dim x z_dim - 1
    determinants = []
    for k in range(z_dim):
        tmp = torch.cat((Ws_new[:, :k], Ws_new[:, k+1:]), dim=1).permute(0, 2, 1).contiguous()
        tmp = tmp.cpu()
        det = torch.det(tmp)
        det = det.cuda()
        if k % 2 == 1:
            det = det * -1
        determinants.append(det)
    determinants = torch.stack(determinants, dim=-1).unsqueeze(-1) # b x z_dim x 1
    determinants = determinants / torch.norm(determinants, dim=1, keepdim=True)
    Ws_new = torch.cat((Ws_new, determinants), dim=-1)
    return Ws_new


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[]):
        super().__init__()
        model = []
        prev_h = input_size
        for h in hidden_sizes + [output_size]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop() # Pop last ReLU
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= n_bit ** 2 - 1 # [0, 15] for n_bit = 4
    x = torch.floor(x + 1e-4) # [0, 15]
    return x

class Decoder(nn.Module):
    prefix = 'decoder'

    def __init__(self, z_dim, channel_dim, discrete=False, n_bit=4):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim
        self.discrete = discrete
        self.n_bit = n_bit
        self.discrete_dim = 2 ** n_bit

        out_dim = self.discrete_dim * self.channel_dim if discrete else channel_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, out_dim, 4, 2, 1),
        )

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        output = self.main(x)

        if self.discrete:
            output = output.view(output.shape[0], self.discrete_dim,
                                 self.channel_dim, *output.shape[2:])
        else:
            output = torch.tanh(output)

        return output


    def loss(self, x, z):
        recon = self(z)
        if self.discrete:
            loss = F.cross_entropy(recon, quantize(x, self.n_bit).long())
        else:
            loss = F.mse_loss(recon, x)
        return loss


    def predict(self, z):
        recon = self(z)
        if self.discrete:
            recon = torch.max(recon, dim=1)[1].float()
            recon = (recon / (self.discrete_dim - 1) - 0.5) / 0.5
        return recon


class InverseModel(nn.Module):
    prefix = 'inv'

    def __init__(self, z_dim, action_dim):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)


class ForwardModel(nn.Module):
    prefix = 'forward'

    def __init__(self, z_dim, action_dim, mode='linear'):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim

        if mode == 'linear':
            self.model = nn.Linear(z_dim + action_dim, z_dim, bias=False)
        else:
            self.model = nn.Sequential(
                nn.Linear(z_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, z_dim),
            )

    def forward(self, z, action):
        x = torch.cat((z, action), dim=1)
        return self.model(x)


class PixelForwardModel(nn.Module):
    def __init__(self, obs_dim, action_dim, learn_delta=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learn_delta = learn_delta

        self.conv1 = nn.Conv2d(obs_dim[0], 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv6 = nn.Conv2d(256, 256, 4, 2, 1)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)

        self.fc1 = nn.Linear(action_dim, 256)
        self.fc2 = nn.Linear(action_dim, 128)
        self.fc3 = nn.Linear(action_dim, 64)
        self.fc4 = nn.Linear(action_dim, 64)

    def forward(self, obs, actions):
        out = F.leaky_relu(self.conv1(obs), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        h1 = F.leaky_relu(self.conv3(out), 0.2) # 32 x 32
        h2 = F.leaky_relu(self.conv4(h1), 0.2) # 16 x 16
        h3 = F.leaky_relu(self.conv5(h2), 0.2) # 8 x 8
        out = F.leaky_relu(self.conv6(h3), 0.2) # 4 x 4

        out = F.leaky_relu(self.deconv1(out) * self.fc1(actions).unsqueeze(-1).unsqueeze(-1), 0.2) + h3
        out = F.leaky_relu(self.deconv2(out) * self.fc2(actions).unsqueeze(-1).unsqueeze(-1), 0.2) + h2
        out = F.leaky_relu(self.deconv3(out) * self.fc3(actions).unsqueeze(-1).unsqueeze(-1), 0.2) + h1
        out = F.leaky_relu(self.deconv4(out) * self.fc4(actions).unsqueeze(-1).unsqueeze(-1))
        out = self.out_conv(out)

        if self.learn_delta:
            out = obs + out
        return out

