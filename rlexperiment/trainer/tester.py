from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_model(model, env, n_eval_episodes=1, deterministic=False):
    """
    评估模型的性能。

    :param model: 要评估的模型
    :param env: 评估环境
    :param n_eval_episodes: 评估的回合数
    :param deterministic: 是否使用确定性策略
    :return: 平均奖励和标准差
    """
    # 加载模型并评估
    loaded_model = model
    mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=n_eval_episodes, deterministic=deterministic)
    
    # 打印结果
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    print('*'*66)
    
    # 返回结果
    return mean_reward, std_reward