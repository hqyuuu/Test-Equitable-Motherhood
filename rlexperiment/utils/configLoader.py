import json
from pathlib import Path

def load_config(config_path):
    # 确保路径是一个Path对象
    config_path = Path(config_path)
    
    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 读取JSON文件并解析为Python对象
    # # with open(config_path, 'r') as file:
    #     config = json.load(file)

    # with open(config_path, 'r', encoding='utf-8') as file:
    # #     config = json.load(file)
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)        
    
    return config

# 示例用法
if __name__ == "__main__":
    # config_path = "test_config.json"
    config_path = "../configs/config_xiantao.json"


    config = load_config(config_path)
    print(config['real_world'])
