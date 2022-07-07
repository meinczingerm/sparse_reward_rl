import os.path

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import Actor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from demonstration.policies.parameterized_reach.policy import ParameterizedReachDemonstrationPolicy
from env.parameterized_reach import ParameterizedReachEnv
from train import _collect_demonstration
from utils import get_project_root_path, save_result_gif


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
        obs_dict = {
            "observation": self.demonstrations["observations"][idx].float(),
            "achieved_goal": self.demonstrations["observations"][idx].float(),
            "desired_goal": self.demonstrations["desired_goals"][idx].float()
        }
        actions = self.demonstrations["actions"][idx].float()
        return obs_dict, actions


class BCModule(pl.LightningModule):
    def __init__(self, policy: Actor):
        super().__init__()
        self.policy = policy

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        observation, action = batch
        predicted_action = self.policy.forward(observation, deterministic=True)
        loss = nn.functional.mse_loss(predicted_action, action)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def _get_model():
    model = TQC(env=ParameterizedReachEnv(number_of_waypoints=1), policy="MultiInputPolicy", policy_kwargs={"net_arch": [512, 512, 512],
                                                                                       "n_critics": 2})
    return model

def train_bc():
    model = _get_model()
    policy = model.actor
    bc_model = BCModule(policy)

    # expert_policy = ParameterizedReachDemonstrationPolicy()
    # env = ParameterizedReachEnv(number_of_waypoints=1)
    demo_path = "/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/ParameterizedReach_1Waypoint/1657201198_045844/demo.hdf5"
    dataset = DemonstrationDataset(demo_path)
    train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    logger = TensorBoardLogger(os.path.join(get_project_root_path(), "training_logs"), name="BC_model")
    trainer = pl.Trainer(max_epochs=10000, accelerator="gpu", logger=logger)
    eval_env = ParameterizedReachEnv(number_of_waypoints=1, has_offscreen_renderer=True)
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])
    # print("Evaluating...")
    # mean_reward, std_reward = evaluate_policy(model, eval_env)
    # print(f"Eval results: Mean:{mean_reward}, Std:{std_reward}")

    trainer.fit(model=bc_model, train_dataloaders=train_dataloader)
    save_result_gif(eval_env, model, trainer.log_dir, "result.gif", 1000)
    print("Evaluating...")
    mean_reward, std_reward = evaluate_policy(model, eval_env)
    print(f"Eval results: Mean:{mean_reward}, Std:{std_reward}")

    print("Done")


if __name__ == '__main__':
    train_bc()




