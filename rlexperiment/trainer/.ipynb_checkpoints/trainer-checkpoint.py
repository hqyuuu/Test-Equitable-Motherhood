import time
from datetime import datetime
from stable_baselines3 import PPO
import json
import torch

def train_and_save_model(env, config, project_path, research_area_lower):
    """
    训练模型并保存模型和配置文件。

    :param env: 环境对象
    :param config: 配置字典
    :param project_path: 项目路径
    :param research_area_lower: 研究领域的名称（小写）
    """
    # 从配置中获取 batch_size，默认值为 64
    batch_size = config.get("batch_size", 64)
    print(f"batch_zise:        {batch_size}")
    congig_beta = config.get("the_2SFCA_beta")
    config_file_name = (config.get("config_file_name") or "").replace(".json", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Andy - PPO runs better on cpu
    print(f"you are using {device} ")
    time.sleep(5)
    # 定义模型
    model = PPO("MlpPolicy", env, gamma=1.0, verbose=1, batch_size=batch_size, device=device)

    total_timesteps_setting = config.get("total_timesteps_setting", 100000)
    start_training_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # 格式化输出当前日期和时间
    start_time = time.time()  # 记录开始时间

    # 训练模型
    model.learn(total_timesteps=total_timesteps_setting)

    # 保存模型
    formatted_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")  # 格式化输出当前日期和时间
    model_path = f'{project_path}model/ppo_{config_file_name}_environment_trained_{total_timesteps_setting}_{formatted_date}.zip'

    # 保存配置文件
    config_save_path = f"{project_path}model/config_ppo_{config_file_name}_environment_trained_{total_timesteps_setting}_{formatted_date}.json"
    print(f'config file saved to {config_save_path}')
    # 定义你不想保存的键
    keys_to_exclude = ["df_od_matrix", "df_Origins", "df_Destinations", "action_space", "observation_space"]

    # 创建一个新的字典，只包含你想要保存的键值对
    # 使用dict.get()方法避免KeyError
    # 对于config中不存在的键，提供默认值"Not provided"
    filtered_config = {key: config.get(key, "Not provided") for key in config if key not in keys_to_exclude}

    # 打印filtered_config，使用json.dumps()确保格式一致
    print(json.dumps(filtered_config, indent=4, ensure_ascii=False))

    # 将filtered_config写入文件
    try:
        with open(config_save_path, 'w') as json_file:
            json.dump(filtered_config, json_file, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"无法写入文件：{e}")

    model.save(model_path)
    end_time = time.time()  # 记录结束时间
    print(f'model trained from {start_training_time} to {formatted_date}, used {end_time - start_time} seconds')
    print(f'model saved to {model_path}')
    return model

# 示例调用
# train_and_save_model(env, config, project_path, research_area_lower)