import os.path
from multiprocessing import Pool

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from demonstration.collect import gather_demonstrations
from demonstration.policies.gridworld.grid_pick_and_place_policy import GridPickAndPlacePolicy
from demonstration.policies.parameterized_reach.fixed_parameterized_reach_policy import FixedParameterizedReachDemonstrationPolicy
from demonstration.policies.parameterized_reach.parameterized_reach_policy import ParameterizedReachDemonstrationPolicy
from env.grid_world_envs.fixed_pick_and_place import FixedGridPickAndPlace
from env.grid_world_envs.pick_and_place import GridPickAndPlace
from env.robot_envs.fixed_parameterized_reach import FixedParameterizedReachEnv
from env.robot_envs.parameterized_reach import ParameterizedReachEnv
from utils import get_project_root_path, save_result_gif, save_dict


class DemonstrationDataset(Dataset):
    """Supervised dataset containing demonstration data"""
    def __init__(self, demonstration_hdf5):
        """
        Init.
        :param demonstration_hdf5: Path to demonstration .hdf5 file.
        """
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
    def __init__(self, model, eval_env, optimizer_kwargs):
        """
        :param model: RL model containing the actor network (necessary for evaluating in environment)
        :param eval_env: evaluation environment
        """
        super().__init__()
        self.model = model
        self.policy = model.actor
        self.eval_env = eval_env
        self.optimizer_kwargs = optimizer_kwargs

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
        rewards, lengths = evaluate_policy(self.model, self.eval_env, n_eval_episodes=20,
                                                  return_episode_rewards=True, deterministic=True)
        success_rate = sum(rewards)/len(rewards)
        self.log("eval/mean_reward", success_rate)
        self.log("eval/mean_ep_length", sum(lengths)/len(lengths))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.optimizer_kwargs)
        return optimizer


def _get_model(_config):
    """
    Get TQC model with config. Later on the policy will be extracted as the trained model.
    Note: data normalization process is the same as used by the TQC policy.
    """
    model = TQC(env=_config['env_class'](**_config["env_kwargs"]), policy="MultiInputPolicy",
                policy_kwargs={**_config["model"]["policy_kwargs"]})
    return model


def train_bc(_config):
    eval_env = _config["env_class"](**_config['env_kwargs'])
    model = _get_model(_config)
    bc_model = BCModule(model, eval_env, _config["model"]["optimizer_kwargs"])

    env = _config["env_class"](**_config['env_kwargs'])
    if _config["regenerate_demonstrations"]:
        assert (_config["training_demo_path"] is None) and (_config["validation_demo_path"] is None)
        expert_policy = _config["expert_policy"]
        if hasattr(expert_policy, "add_env"):
            expert_policy.add_env(env)
        training_demo_path = gather_demonstrations(env, demonstration_policy=expert_policy,
                                                   episode_num=_config["number_of_demonstrations"])
        validation_demo_path = gather_demonstrations(env, demonstration_policy=expert_policy,
                                                     episode_num=40)
    else:
        training_demo_path = _config["training_demo_path"]
        validation_demo_path = _config["validation_demo_path"]
    dataset = DemonstrationDataset(training_demo_path)
    train_dataloader = DataLoader(dataset, batch_size=_config["model"]["batch_size"], shuffle=True)
    val_dataset = DemonstrationDataset(validation_demo_path)
    val_dataloader = DataLoader(val_dataset, batch_size=_config["model"]["batch_size"])

    logger = TensorBoardLogger(os.path.join(get_project_root_path(), "training_logs"),
                               name=f"BC_model_{env.name}")
    trainer = pl.Trainer(max_epochs=50000, accelerator="gpu", logger=logger, check_val_every_n_epoch=5)
    save_dict(_config, os.path.join(trainer.log_dir, "config.json"))
    eval_env = _config["env_class"](**_config["env_kwargs"])  #, has_offscreen_renderer=True)
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    trainer.fit(model=bc_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    save_result_gif(eval_env, model, trainer.log_dir, "result.gif", 1000)

    print("Done")


def run_parallel(_configs):
    pool = Pool(processes=len(_configs))

    # map the function to the list and pass
    # function and list_ranges as arguments
    pool.map(train_bc, _configs)


def load_and_eval(log_dir, _config, num_of_eval_videos=5):
    """
    Loads back and evaluates the saved model.
    :param log_dir: log_dir path including the saved model
    :param _config: config of the trained model
    :param num_of_eval_videos: number of videos to generate during the evaluation
    :return: None
    """
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

    # Example config
    configs = [
                {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                        "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                      np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                        },
                "env_class": FixedParameterizedReachEnv,
                "number_of_demonstrations": 100,
                "regenerate_demonstrations": True,
                "training_demo_path": None,
                "validation_demo_path": None,
                "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0),
                "use_policy_wrapper": False,
                "model": {"batch_size": 2048,
                          "policy_kwargs": {"net_arch": [32, 32]},
                          "optimizer_kwargs": {"lr": 1e-3}}
                },
                {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                                "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                              np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                                },
                 "env_class": FixedParameterizedReachEnv,
                 "number_of_demonstrations": 50,
                 "regenerate_demonstrations": True,
                 "training_demo_path": None,
                 "validation_demo_path": None,
                 "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0),
                 "use_policy_wrapper": False,
                 "model": {"batch_size": 2048,
                           "policy_kwargs": {"net_arch": [32, 32]},
                           "optimizer_kwargs": {"lr": 1e-3}}
                 },
                {"env_kwargs": {"horizon": 300, "number_of_waypoints": 2,
                                "waypoints": [np.array([0.58766, 0.26816, 0.37820, 2.89549, 0.03567, -0.39348]),
                                              np.array([0.42493, 0.07166, 0.36318, 2.88426, 0.12777, 0.35920])]
                                },
                 "env_class": FixedParameterizedReachEnv,
                 "number_of_demonstrations": 10,
                 "regenerate_demonstrations": True,
                 "training_demo_path": None,
                 "validation_demo_path": None,
                 "expert_policy": FixedParameterizedReachDemonstrationPolicy(env=None, randomness_scale=0),
                 "use_policy_wrapper": False,
                 "model": {"batch_size": 2048,
                           "policy_kwargs": {"net_arch": [32, 32]},
                           "optimizer_kwargs": {"lr": 1e-3}}
                 },
               ]

    run_parallel(configs)



