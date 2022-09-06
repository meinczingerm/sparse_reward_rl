import numpy as np
import sklearn.metrics
from sklearn.metrics import confusion_matrix, accuracy_score

from env.goal_handler import HinDRLGoalHandler


def eval_goal_handler(goal_handler: HinDRLGoalHandler, samples_from_each_episode=1000):
    m = goal_handler.epsilon_params["m"]
    true_pairs = []
    false_pairs = []
    for episode_demo in goal_handler.demonstrations:
        demo_len = len(episode_demo)
        for _ in range(samples_from_each_episode):
            idx_1 = np.random.randint(0, demo_len-1)
            obs_1 = episode_demo[idx_1]
            idx_2 = np.clip(np.random.randint(idx_1 - m, idx_1 + m), 0, demo_len-1)
            obs_2 = episode_demo[idx_2]
            true_pairs.append((obs_1, obs_2))

            idx_1 = np.random.randint(0, demo_len-1)
            obs_1 = episode_demo[idx_1]
            possible_indices = np.hstack([np.arange(0, idx_1-m), np.arange(idx_1 + m,  demo_len - 1)])
            idx_2 = np.random.choice(possible_indices, 1)[0]
            obs_2 = episode_demo[idx_2]
            false_pairs.append((obs_1, obs_2))



    y_pred = []
    for pair in true_pairs:
        y_pred.append(goal_handler.compute_reward(pair[0], pair[1], [{}]))
    y_pred_1 = np.array(y_pred)
    y_true_1 = np.ones_like(y_pred_1)

    y_pred = []
    for pair in false_pairs:
        y_pred.append(goal_handler.compute_reward(pair[0], pair[1], [{}]))
    y_pred_2 = np.array(y_pred)
    y_true_2 = np.zeros_like(y_pred_2)

    y_true_labels = np.hstack([y_true_1, y_true_2])
    y_pred = np.hstack([y_pred_1, y_pred_2])

    print(goal_handler.epsilon)
    print(confusion_matrix(y_true_labels, y_pred))
    print(accuracy_score(y_true_labels, y_pred))


if __name__ == '__main__':
    goal_handler = HinDRLGoalHandler("/home/mark/tum/2022ss/thesis/master_thesis/demonstration/collection/GridPickAndPlace/10000_1662034052_3822935/demo.hdf5",
                                     m=2, k=0)
    eval_goal_handler(goal_handler, samples_from_each_episode=2)