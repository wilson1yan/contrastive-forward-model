from typing import Optional, List
import torch
from torch import nn
from torch.nn import functional as F


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class SSM(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, state_size, action_size, hidden_size, embedding_size, trans_type, activation_function='relu'):
        super().__init__()
        self.encoder = VisualEncoder(state_size, embedding_size, activation_function=activation_function)
        self.trans = TransitionModel(state_size, action_size, hidden_size,
                                       activation_function)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, observations: Optional[torch.Tensor] = None) -> \
    List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_states[0], posterior_states[0] = prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            prior_means[t + 1], prior_std_devs[t + 1], prior_states[t + 1] = self.trans.forward_train(_state, actions[t])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                posterior_means[t + 1], posterior_std_devs[t + 1], posterior_states[t + 1] = self.encoder.forward_train(observations[t])
        # Return new hidden states
        hidden = [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0),
                  torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class TransitionModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2 * state_size)
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev

    def forward_train(self, state, action):
        out = torch.cat((state, action), dim=1)
        out = self.act_fn(self.fc1(out))
        means, std_devs = self.fc2(out).chunk(2, dim=1)
        std_devs = F.softplus(std_devs) + self.min_std_dev
        samples = means + torch.randn_like(means) * std_devs
        return means, std_devs, samples

    def forward(self, state, action):
        out = torch.cat((state, action), dim=1)
        out = self.act_fn(self.fc1(out))
        means, std_devs = self.fc2(out).chunk(2, dim=1)
        return means


class CPCTransitionModel(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, activation_function='relu',
                 min_std_dev=0.1, trans_type='linear'):
        super().__init__()
        self.trans_type = trans_type
        print('trans_type', self.trans_type)
        self.state_size = state_size
        self.min_std_dev = min_std_dev
        if self.trans_type == 'linear':
            self.model = nn.Linear(state_size + action_size, 2 * state_size, bias=False)
        elif self.trans_type == 'mlp':
            hidden_size = 64
            self.model = nn.Sequential(
                nn.Linear(state_size + action_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 2 * state_size)
            )
        elif self.trans_type == 'reparam_w':
            self.model = nn.Sequential(
                nn.Linear(state_size + action_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * state_size * state_size)
            )
        else:
            raise Exception('Invalid trans_type', self.trans_type)

    def forward(self, z, a):
        out = torch.cat((z, a), dim=1)
        out = self.model(out)
        if self.trans_type == 'reparam_w':
            Ws = out.view(out.shape[0], 2 * self.state_size, self.state_size)
            out = torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1)
        return out.chunk(2, dim=1)[0]

    def forward_train(self, z, a):
        out = torch.cat((z, a), dim=1)
        out = self.model(out)
        if self.trans_type == 'reparam_w':
            Ws = out.view(out.shape[0], 2 * self.state_size, self.state_size)
            out = torch.bmm(Ws, z.unsqueeze(-1)).squeeze(-1)
        means, std_devs = out.chunk(2, dim=1)
        std_devs = F.softplus(std_devs) + self.min_std_dev
        samples = means + torch.randn_like(means) * std_devs
        return means, std_devs, samples


class VisualObservationModel(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2) # 5 x 5
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)  # 13 x 13
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2) # 30 x 30
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2) # 64 x 64

   # @jit.script_method
    def forward(self, state):
        hidden = self.fc1(state)  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(state_size, embedding_size, activation_function='relu'):
    return VisualObservationModel(state_size, embedding_size, activation_function)


class VisualEncoder(nn.Module):
    __constants__ = ['embedding_size']

    def __init__(self, state_size, embedding_size, activation_function='relu', min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.state_size = state_size
        self.min_std_dev = min_std_dev

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc1 = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 2 * state_size)

    def forward_train(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.act_fn(self.fc1(hidden))  # Identity if embedding size is 1024 else linear projection
        hidden = self.fc2(hidden)
        means, std_devs = hidden.chunk(2, dim=1)
        std_devs = F.softplus(std_devs) + self.min_std_dev
        samples = means + torch.randn_like(means) * std_devs
        return means, std_devs, samples

    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.act_fn(self.fc1(hidden))  # Identity if embedding size is 1024 else linear projection
        hidden = self.fc2(hidden)
        means, std_devs = hidden.chunk(2, dim=1)
        return means


class CPCEncoder(nn.Module):
    prefix = 'encoder'

    def __init__(self, state_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Linear(256 * 4 * 4, 2 * state_size)
        self.min_std_dev = min_std_dev

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        mean, std_devs = x.chunk(2, dim=1)
        return mean

    def forward_train(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        mean, std_devs = x.chunk(2, dim=1)
        std_devs = F.softplus(std_devs) + self.min_std_dev
        samples = mean + std_devs * torch.randn_like(mean)
        return mean, std_devs, samples


def Encoder(state_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    return VisualEncoder(state_size, embedding_size, activation_function, min_std_dev)
