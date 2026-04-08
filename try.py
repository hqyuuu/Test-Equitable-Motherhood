from rlexperiment.rlsolver import main


# Train and test
main(
    datasets_path="./datasets/exported_data",
    config_path="./rlexperiment/configs/Florida/config_Florida_1a_beta_1_0.json"
)

# Test with existing model
main(
    datasets_path="./datasets/exported_data",
    config_path="./rlexperiment/configs/Florida/config_Florida_1a_beta_1_0.json",
    model_path="./model/your_model.zip"
)