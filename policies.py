import random
import torch
from torchvision import transforms
import models


class DQNPolicy:
    def __init__(self, cfg, action_space, train=False, random_seed=None):
        self.cfg = cfg
        self.action_space = action_space
        self.train = train

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = self.build_network()
        self.transform = transforms.ToTensor()

        # Resume from checkpoint if applicable
        if self.cfg.checkpoint_path is not None:
            model_checkpoint = torch.load(self.cfg.model_path, map_location=self.device)
            self.policy_net.load_state_dict(model_checkpoint['state_dict'])
            if self.train:
                self.policy_net.train()
            else:
                self.policy_net.eval()
            print("=> loaded model '{}'".format(self.cfg.model_path))

        if random_seed is not None:
            random.seed(random_seed)

    def build_network(self):
        raise NotImplementedError

    def apply_transform(self, s):
        return self.transform(s).unsqueeze(0)

    def step(self, state, exploration_eps=None, debug=False):
        if exploration_eps is None:
            exploration_eps = self.cfg.final_exploration
        state = self.apply_transform(state).to(self.device)
        with torch.no_grad():
            output = self.policy_net(state).squeeze(0)
        if random.random() < exploration_eps:
            action = random.randrange(self.action_space)
        else:
            action = output.view(1, -1).max(1)[1].item()
        info = {}
        if debug:
            info['output'] = output.squeeze(0)
        return action, info

class SteeringCommandsPolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.SteeringCommandsDQN(num_input_channels=self.cfg.num_input_channels, num_output_channels=self.action_space)
        ).to(self.device)

class DenseActionSpacePolicy(DQNPolicy):
    def build_network(self):
        return torch.nn.DataParallel(
            models.DenseActionSpaceDQN(num_input_channels=self.cfg.num_input_channels)
        ).to(self.device)
