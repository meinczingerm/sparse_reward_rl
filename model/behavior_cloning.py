import os.path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.parameterized_reach import ParameterizedReachEnv
from train import _collect_demonstration
from utils import get_project_root_path


class DemonstrationDataset(Dataset):
    def __init__(self, demonstration_hdf5):
        self.demonstrations = {"observations": [],
                               "desired_goals": [],
                               "actions": []}
        with h5py.File(demonstration_hdf5, "r") as f:
            for demo_id in f["data"].keys():
                actions = np.array(f["data"][demo_id]["actions"])
                observations = np.array(f["data"][demo_id]["observations"])
                desired_goals = np.array(f["data"][demo_id]["desired_goal"])
                self.demonstrations["actions"].append(actions)
                self.demonstrations["observations"].append(observations)
                self.demonstrations["desired_goals"].append(desired_goals)
        for key in self.demonstrations.keys():
            self.demonstrations[key] = torch.from_numpy(np.vstack(self.demonstrations[key]))

    def __len__(self):
        return self.demonstrations["observations"].shape[0]

    def __getitem__(self, idx):
        stacked_obs_with_goal = torch.cat([self.demonstrations["observations"][idx],
                                           self.demonstrations["desired_goals"][idx]]).float()
        actions = self.demonstrations["actions"][idx].float()
        return stacked_obs_with_goal, actions


class BCModule(pl.LightningModule):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        observation, action = batch
        predicted_action = self.policy(observation)
        loss = nn.functional.mse_loss(predicted_action, action)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    # expert_policy = ParameterizedReachDemonstrationPolicy()
    # env = ParameterizedReachEnv()

    # demo_path = _collect_demonstration(env, expert_policy, 5)
    demo_path = "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_2Waypoint/1657026234_7696974/demo.hdf5"
    dataset = DemonstrationDataset(demo_path)
    train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    policy = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 7))
    bc_model = BCModule(policy)

    logger = TensorBoardLogger(os.path.join(get_project_root_path(),"training_logs"), name="BC_model")
    trainer = pl.Trainer(max_epochs=10000, accelerator="gpu", logger=logger)
    trainer.fit(model=bc_model, train_dataloaders=train_dataloader)





