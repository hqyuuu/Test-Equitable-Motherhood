import gymnasium as gym
import numpy as np

def setup_environment(config):
    # 定义要打印的键
    keys_to_print = [
        "research_area", "real_world", "max_Supply", "max_demand", "target_ratio",
        "the_2SFCA_beta", "total_timesteps_setting", "max_steps_per_experiment",
        "max_amount_per_transfer", "qp_like", "max_distance","remain_origin_supply","Punishment_for_Violating_Conditions",
        "additional_info", "areas_need_focus"
    ]
    
    # 创建一个新的字典，只包含要保存和打印的键值对
    # filtered_config = {key: config[key] for key in keys_to_print}
    # 使用get()方法避免KeyError
    filtered_config = {key: config.get(key, None) for key in keys_to_print}
    
    # 打印特定的键值对
    for key, value in filtered_config.items():
        # 如果值是列表，将其转换为字符串
        if isinstance(value, list):
            value = str(value)
        # 如果值是None，使用一个默认的字符串
        elif value is None:
            value = "Not provided"
        print(f"{key:<40}: {value:<40}")
    
    # 根据配置参数确定供给点(如：医院)数量
    if config["real_world"]:
        num_Destinations = len(config["df_Destinations"])
        num_Origins = len(config["df_Origins"])
        print(f'{"number of Supply point(Dest):":<40}{num_Destinations:<40}"\n{"number of Demand Points (Origin):":<40}{num_Origins:<40}')
    else:
        # 随机创建医院
        num_Destinations = 5  # 默认供给点(如：医院)数量
        num_Origins = 10
        print('随机创建医院')
        print(f'{"number of Supply point(Dest):":<40}{num_Destinations:<40}"\n{"number of Demand Points (Origin):":<40}{num_Origins:<40}')
    
    # 定义动作空间
    max_amount_per_transfer = config.get("max_amount_per_transfer", 1000)
    action_space = gym.spaces.MultiDiscrete([num_Destinations, num_Destinations, max_amount_per_transfer])
    
    # 观察空间(下面几行不参与实际运行)
    # 供给点(如：医院)观察维度：序号(1) + 经度(1) + 纬度(1) + 劳动力总量(1)
    Destination_observation_dims = 4
    # 需求点(如：居民区)观察维度：经度(1) + 纬度(1) + 人口(1) + 可达性(1)
    Originial_observation_dims = 4
    # 总观察维度(相乘)
    total_Destination_observation_dims = num_Destinations * Destination_observation_dims
    total_Originial_observation_dims = num_Origins * Originial_observation_dims
    observation_dims = total_Destination_observation_dims + total_Originial_observation_dims
    
    # 每个属性的最大值和最小值
    Destination_min_vals = np.array([0, -180, -90, 0])  # 供给点(如：医院)属性的最小值
    Destination_max_vals = np.array([num_Destinations-1, 180, 90, config["max_Supply"]])  # 供给点(如：医院)属性的最大值
    
    Originial_min_vals = np.array([-180, -90, 0, 0])  # 需求点(如：居民区)属性的最小值
    Originial_max_vals = np.array([180, 90, config["max_demand"], 1000])  # 需求点(如：居民区)属性的最大值
    
    # 创建观察空间的最小值和最大值数组
    low = np.tile(Destination_min_vals, num_Destinations).tolist() + np.tile(Originial_min_vals, num_Origins).tolist()
    high = np.tile(Destination_max_vals, num_Destinations).tolist() + np.tile(Originial_max_vals, num_Origins).tolist()
    
    # 定义观察空间
    observation_space = gym.spaces.Box(
        low=np.array(low, dtype=np.float32),
        high=np.array(high, dtype=np.float32),
        dtype=np.float32
    )
    
    return action_space, observation_space
if __name__ == "__main__":
    # 示例调用
    config = {
        "research_area": "example",
        "real_world": True,
        "max_Supply": 1000,
        "max_demand": 2000,
        "target_ratio": 0.5,
        "the_2SFCA_beta": 0.1,
        "total_timesteps_setting": 100,
        "max_steps_per_experiment": 50,
        "max_amount_per_transfer": 500,
        "qp_like": True,
        "Punishment_for_Violating_Conditions": 10,
        "additional_info": "some info",
        "areas_need_focus": ["area1", "area2"],
        "df_Destinations": [1, 2, 3],
        "df_Origins": [4, 5, 6, 7]
    }

    action_space, observation_space = setup_environment(config)
    print("Action Space:", action_space)
    print("Observation Space:", observation_space)