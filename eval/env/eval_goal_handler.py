import numpy as np
import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score

from env.goal_handler import HinDRLGoalHandler


def eval_goal_handler(goal_handler: HinDRLGoalHandler, samples_from_each_episode=1000):
    m = goal_handler.epsilon_params["m"]
    true_pairs = ([], [])
    false_pairs = ([], [])
    for episode_demo in goal_handler.demonstrations:
        demo_len = len(episode_demo)
        for _ in range(samples_from_each_episode):
            idx_1 = np.random.randint(0, demo_len-1)
            obs_1 = episode_demo[idx_1]
            idx_2 = np.clip(np.random.randint(idx_1 - m, idx_1 + m), 0, demo_len-1)
            obs_2 = episode_demo[idx_2]
            true_pairs[0].append(obs_1)
            true_pairs[1].append(obs_2)

            idx_1 = np.random.randint(0, demo_len-1)
            obs_1 = episode_demo[idx_1]
            possible_indices = np.hstack([np.arange(0, idx_1-m), np.arange(idx_1 + m,  demo_len - 1)])
            idx_2 = np.random.choice(possible_indices, 1)[0]
            obs_2 = episode_demo[idx_2]
            false_pairs[0].append(obs_1)
            false_pairs[1].append(obs_2)

    true_achieved = np.vstack(true_pairs[0])
    true_goals = np.vstack([true_pairs[1]])
    y_pred_1 = goal_handler.compute_reward(true_achieved, true_goals, [{}])
    y_true_1 = np.ones_like(y_pred_1)

    false_achieved = np.vstack(false_pairs[0])
    false_goals = np.vstack(false_pairs[1])
    y_pred_2 = goal_handler.compute_reward(false_achieved, false_goals, [{}])
    y_true_2 = np.zeros_like(y_pred_2)

    y_true_labels = np.hstack([y_true_1, y_true_2])
    y_pred = np.hstack([y_pred_1, y_pred_2])

    print(goal_handler.epsilon)
    print(confusion_matrix(y_true_labels, y_pred))
    print(accuracy_score(y_true_labels, y_pred))


if __name__ == '__main__':
    goal_handler = HinDRLGoalHandler("/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/GridPickAndPlace/100_1663144129_5172603/demo.hdf5",
                                     m=1, k=0)
    eval_goal_handler(goal_handler, samples_from_each_episode=2)