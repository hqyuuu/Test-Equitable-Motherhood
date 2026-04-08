from rlexperiment.rlsolver import main
import os

model_dir = "./model"

files = sorted(
    [f for f in os.listdir(model_dir) if f.endswith(".zip")],
    key=lambda x: os.path.getmtime(os.path.join(model_dir, x))
)

if len(files) == 0:
    raise ValueError("No model found!")

latest_model = files[-1]
model_path = os.path.join(model_dir, latest_model.replace(".zip", ""))

print("Loading model:", model_path)

main(
    datasets_path="./datasets/exported_data",
    config_path="./rlexperiment/configs/Florida/config_Florida_1a_beta_1_0.json",
    model_path=model_path
)