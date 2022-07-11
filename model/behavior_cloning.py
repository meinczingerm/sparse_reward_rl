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
    def __init__(self, model, eval_env):
        """
        :param model: RL model containing the actor network (necessary for evaluating in environment)
        :param eval_env: evaluation environment
        """
        super().__init__()
        self.model = model
        self.policy = model.actor
        self.eval_env = eval_env

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        observation, action = batch
        predicted_action = self.policy.forward(observation, deterministic=True)
        loss = nn.functional.mse_loss(predicted_action, action)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        observation, action = batch
        predicted_action = self.policy.forward(observation, deterministic=True)
        loss = nn.functional.mse_loss(predicted_action, action)
        # Logging to TensorBoard by default
        self.log("validation_loss", loss)

    def validation_epoch_end(self, *args) -> None:
        rewards, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=20,
                                                  return_episode_rewards=True)
        success_rate = sum(rewards)/len(rewards)
        self.log("env_success_rate", success_rate)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def _get_model(_config):
    model = TQC(env=ParameterizedReachEnv(**_config["env_kwargs"]), policy="MultiInputPolicy",
                policy_kwargs={**_config["model"]["policy_kwargs"]})
    return model


def train_bc(_config):
    eval_env = ParameterizedReachEnv(**_config['env_kwargs'])
    model = _get_model(_config)
    bc_model = BCModule(model, eval_env)

    if _config["regenerate_demonstrations"]:
        assert (_config["training_demo_path"] is None) and (_config["validation_demo_path"] is None)
        expert_policy = _config["expert_policy"]
        env = ParameterizedReachEnv(**_config['env_kwargs'])
        training_demo_path = _collect_demonstration(env, demonstration_policy=expert_policy,
                                           episode_num=_config["number_of_demonstrations"])
        validation_demo_path = _collect_demonstration(env, demonstration_policy=expert_policy,
                                           episode_num=_config["number_of_demonstrations"])
    else:
        training_demo_path = _config["training_demo_path"]
        validation_demo_path = _config["validation_demo_path"]
    dataset = DemonstrationDataset(training_demo_path)
    train_dataloader = DataLoader(dataset, batch_size=_config["model"]["batch_size"], shuffle=True)
    val_dataset = DemonstrationDataset(validation_demo_path)
    val_dataloader = DataLoader(val_dataset, batch_size=_config["model"]["batch_size"])

    logger = TensorBoardLogger(os.path.join(get_project_root_path(), "training_logs"),
                               name=f"BC_model_{env.name}")
    trainer = pl.Trainer(max_epochs=50000, accelerator="gpu", logger=logger, check_val_every_n_epoch=2000)
    eval_env = ParameterizedReachEnv(**_config["env_kwargs"], has_offscreen_renderer=True)
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    trainer.fit(model=bc_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    save_result_gif(eval_env, model, trainer.log_dir, "result.gif", 1000)

    print("Done")


def load_and_eval(log_dir, _config, num_of_eval_videos=5):
    checkpoint_path = os.path.join(log_dir, "checkpoints")
    file_name = os.listdir(checkpoint_path)[0]
    checkpoint_path = os.path.join(checkpoint_path, file_name)

    model = _get_model(_config)

    model = BCModule.load_from_checkpoint(checkpoint_path, model=model, eval_env=None).model

    eval_env = _config['env_class'](**_config["env_kwargs"], has_offscreen_renderer=True)
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    for i in range(num_of_eval_videos):
        save_result_gif(eval_env, model, log_dir, f"result{i}.gif", 1000)


if __name__ == '__main__':
    config = {"env_kwargs": {"number_of_waypoints": 2},
              "env_class": ParameterizedReachEnv,
              "number_of_demonstrations": 30,
              "regenerate_demonstrations": True,
              "training_demo_path": None,
              "validation_demo_path": None,
              "expert_policy": ParameterizedReachDemonstrationPolicy(),
              "model": {"batch_size": 1024,
                        "policy_kwargs": {"net_arch": [64]}}}

    train_bc(config)




