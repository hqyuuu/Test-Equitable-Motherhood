#!/bin/bash
echo "Starting"

# 定义起始和结束的 beta 值
start_beta=0.6
end_beta=2.0
step=0.2

# 定义运行实验的函数
run_experiments() {
    local city=$1
    local config_prefixes=("${!2}")  # 使用间接引用获取数组
    local config_dir="./rlexperiment/configs/${city}"
    local dataset_dir="./datasets/exported_data"

    for prefix in "${config_prefixes[@]}"; do
        echo "Running experiments for ${city}-${prefix}"

        # 初始化 beta 值（强制格式化为 0.x）
        beta=$(printf "%.1f" "$start_beta")

        # 循环生成实验命令
        while (( $(echo "$beta <= $end_beta" | bc -l) )); do
            # 格式化 beta 值为固定格式 (e.g., 0.8 → 0_8)
            beta_str=$(LC_ALL=C printf "%.1f" "$beta" | tr '.' '_')

            # 生成配置文件名和日志文件名
            config_file="${config_dir}/config_${city}_${prefix}_beta_${beta_str}.json"
            log_file="output_${city}_${prefix}_beta_${beta_str}.log"

            # 检查配置文件是否存在
            if [ ! -f "$config_file" ]; then
                echo "Error: Config file $config_file not found! Skipping..."
                beta=$(LC_ALL=C printf "%.1f" "$(echo "$beta + $step" | bc -l)")
                continue
            fi

            # 运行实验
            echo "Starting experiment: beta=$beta, config=$config_file"
            nohup python -m rlexperiment.rlsolver -d "$dataset_dir" -c "$config_file" > "$log_file" 2>&1 &

            # 更新 beta 值（强制格式化，避免 .8）
            beta=$(LC_ALL=C printf "%.1f" "$(echo "$beta + $step" | bc -l)")
        done
    done
}


echo "Florida"
florida_prefixes=("1a")

run_experiments "Florida" florida_prefixes[@]


echo "All experiments started in the background."