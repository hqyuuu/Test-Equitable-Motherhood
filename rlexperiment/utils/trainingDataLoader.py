import os
import pandas as pd

def load_training_data(research_area, dataset_dir="../datasets"):
    """
    根据研究区名称加载OD矩阵、起点和终点数据。
    
    参数:
    research_area (str): 研究区名称 (如 'xiantao', 'chicago', 'columbus')。
    dataset_dir (str): 数据集文件夹的路径，默认是 '../datasets'。
    
    返回:
    tuple: 包含OD矩阵、起点和终点的DataFrame。
    """
    # 将研究区名称转换为小写
    research_area_lower = research_area.lower()

    # 定义文件路径
    od_matrix_file = os.path.join(dataset_dir, f'{research_area_lower}_od_matrix.csv')
    origins_file = os.path.join(dataset_dir, f'{research_area_lower}_origins.csv')
    destinations_file = os.path.join(dataset_dir, f'{research_area_lower}_destinations.csv')

    # 检查文件是否存在
    if not os.path.exists(od_matrix_file) or not os.path.exists(origins_file) or not os.path.exists(destinations_file):
        raise FileNotFoundError(f"File not found, please check path: {od_matrix_file}, {origins_file}, {destinations_file}")
    
    # 读取CSV文件到DataFrame
    od_matrix = pd.read_csv(od_matrix_file)
    origins = pd.read_csv(origins_file)
    destinations = pd.read_csv(destinations_file)

    print(f"Loading Success: {research_area} Training Data.")
    return od_matrix, origins, destinations


if __name__ == '__main__':
    load_training_data('xiantao',dataset_dir='../../datasets/exported_data')