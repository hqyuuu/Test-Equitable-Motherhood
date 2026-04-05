# Maternal Healthcare Accessibility Optimization with Reinforcement Learning
test
## Table of Contents
- [Citation](#citation)
- [About the Project](#project-overview)
- [Code Usage](#usage)
- [Folder Structure](#project-structure)
- [Contact](#contact)

## Citation

If you use this algorithm in your research or applications, please cite this source:

```
@article{qin2025maternal},
journal = {ACM SIGSPATIAL},
title = {Leveraging Reinforcement Learning for Maternity Care Resource Reallocation: A Case Study in Florida},
year = {2025},
author = {Qin, Andy and Kang, Yuhao and Li, Hanqi and Wang, Shiqi and Zhou, Bing and Wang, Fahui and Hung, Peiyin},
keywords = {Spatial Optimization, Reinforcement Learning, Public Health, Maximal Accessibility Equality Problem (MAEP)},
}
```

## Project Overview

This project addresses maternal healthcare accessibility optimization by using reinforcement learning to determine optimal resource allocation (specifically OBBD - Obstetric Beds and Delivery services) between hospitals. The system uses the 2SFCA method to calculate accessibility scores and employs PPO to learn strategies that minimize accessibility variance while respecting distance constraints.

### Key Features

- **Real-world healthcare optimization**: Uses actual Florida maternal health data
- **2SFCA accessibility calculation**: Implements gravity-based accessibility models
- **Reinforcement Learning**: Uses PPO for policy optimization
- **Distance constraints**: Enforces realistic transfer distance limits
- **Multiple scenarios**: Supports various optimization objectives (distance, accessibility caps, underserved communities)
- **Performance optimization**: CPU/GPU training support with performance logging

## Requirements

### System Requirements
- Python 3.8+

### Python Dependencies

Create a virtual environment and install the following packages:

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Geospatial and accessibility
geopandas>=0.10.0
geopy>=2.2.0
shapely>=1.8.0

# Machine Learning and RL
stable-baselines3>=1.6.0
gymnasium>=0.28.0
torch>=1.12.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
pygame>=2.1.0  # For visualization support
IPython>=7.0.0
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Maternal-Access-RL
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

### Required Dataset Files

The project requires three CSV files in the `datasets/exported_data/` directory. All data have been aggregated, and statistical noise has been added to ensure no individual-information can be detected:

#### 1. Origins File (`{research_area}_origins.csv`)
Contains demand points (e.g., residential areas) with columns:
- `OriginID`: Unique identifier for each origin
- `O_Demand`: Population/demand at each origin
- `lng`: Longitude coordinate
- `lat`: Latitude coordinate

#### 2. Destinations File (`{research_area}_destinations.csv`)
Contains supply points (e.g., hospitals) with columns:
- `DestinationID`: Unique identifier for each destination
- `D_Supply`: Supply capacity (e.g., number of OBBD beds)
- `lng`: Longitude coordinate
- `lat`: Latitude coordinate

#### 3. OD Matrix File (`{research_area}_od_matrix.csv`)
Contains travel costs between origins and destinations:
- `OriginID`: Origin identifier
- `DestinationID`: Destination identifier
- `TravelCost`: Travel cost/distance between origin and destination in Km
- `orig_origin`: Original origin identifier

### Example Dataset Structure

For the Florida dataset, files should be named:
- `florida_origins.csv`
- `florida_destinations.csv`
- `florida_od_matrix.csv`

## Configuration

### Configuration File Structure

Create JSON configuration files in `rlexperiment/configs/{research_area}/` with the following structure:

```json
{
    "research_area": "Florida",
    "real_world": true,
    "max_Supply": 100000,
    "max_demand": 100000,
    "target_ratio": 0.3,
    "the_2SFCA_beta": 1.0,
    "total_timesteps_setting": 100000,
    "max_amount_per_transfer": 100,
    "Punishment_for_Violating_Conditions": 1000,
    "qp_like": false,
    "remain_origin_supply": true,
    "max_distance": 50000,
    "max_steps_per_experiment": 1000,
    "areas_need_focus": [17, 18, 31, 43],
    "config_file_name": "config_Florida_1a_beta_1_0.json"
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `research_area` | Name of the research area | Required |
| `real_world` | Use real-world data (vs. synthetic) | true |
| `max_Supply` | Maximum supply capacity | 100000 |
| `max_demand` | Maximum demand capacity | 100000 |
| `target_ratio` | Target variance reduction ratio | 0.3 |
| `the_2SFCA_beta` | Distance decay parameter for 2SFCA | 1.0 |
| `total_timesteps_setting` | Training timesteps | 100000 |
| `max_amount_per_transfer` | Maximum units per transfer | 100 |
| `max_distance` | Maximum transfer distance (meters) | 50000 |
| `max_steps_per_experiment` | Maximum steps per episode | 1000 |
| `areas_need_focus` | List of focus area IDs | [] |

## Usage

### Basic Training and Testing

1. **Train a new model:**
```bash
python -m rlexperiment.rlsolver -d ./datasets/exported_data -c ./rlexperiment/configs/Florida/config_Florida_1a_beta_1_0.json
```

2. **Test with existing model:**
```bash
python -m rlexperiment.rlsolver -d ./datasets/exported_data -c ./rlexperiment/configs/Florida/config_Florida_1a_beta_1_0.json --model_path ./model/ppo_config_Florida_1a_beta_1_0_environment_trained_100000_2024_01_01_12_00_00.zip
```

### Batch Experiments

Run multiple experiments with different parameters:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This script runs experiments with different beta values (0.6 to 2.0) for the 2SFCA method.

### Programmatic Usage

```python
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
```

## Project Structure

```
Maternal-Access-RL/
├── rlexperiment/                 # Main RL experiment package
│   ├── envs/                     # RL environment implementations
│   │   └── env_v3.py            # Main environment (v3 - optimized)
│   ├── trainer/                  # Training and evaluation modules
│   │   ├── trainer.py           # Model training logic
│   │   └── tester.py            # Model evaluation
│   ├── utils/                    # Utility functions
│   │   ├── spatialAccessibility.py  # 2SFCA calculations
│   │   ├── trainingDataLoader.py    # Data loading
│   │   ├── configLoader.py          # Configuration management
│   │   └── configProcessor.py       # Environment setup
│   ├── configs/                  # Configuration files
│   │   └── Florida/             # Florida-specific configs
│   └── rlsolver.py              # Main solver script
├── datasets/                     # Data storage
│   ├── exported_data/           # Processed datasets
│   └── orig_datasets/           # Original datasets
├── model/                       # Trained models storage
├── Output/                      # Experiment outputs
├── run_experiments.sh          # Batch experiment script
└── README.md                   # This file
```

## Algorithm Details

### 2SFCA (Two-Step Floating Catchment Area)

The system uses the 2SFCA method to calculate healthcare accessibility:

1. **Step 1**: Calculate supply-to-demand ratio for each healthcare facility
2. **Step 2**: Sum weighted ratios for each demand point based on distance decay

The gravity model is used with the formula:
```
Accessibility_i = Σ(S_j / Σ(D_k * d_kj^(-β))) * d_ij^(-β)
```

Where:
- `S_j`: Supply at destination j
- `D_k`: Demand at origin k
- `d_ij`: Distance between origin i and destination j
- `β`: Distance decay parameter

### Reinforcement Learning Environment

- **State Space**: Coordinates, supply, demand, and accessibility of all locations
- **Action Space**: MultiDiscrete [start_hospital, end_hospital, transfer_amount]
- **Reward Function**: Based on accessibility variance reduction with distance penalties
- **Constraints**: Distance limits, supply availability, accessibility caps

## Output Files

### Model Files
- Trained models: `model/ppo_{config_name}_environment_trained_{timesteps}_{timestamp}.zip`
- Configuration backups: `model/config_ppo_{config_name}_environment_trained_{timesteps}_{timestamp}.json`

### Experiment Results
- Origins results: `Output/{research_area}_{beta}_Origins_{timestamp}.csv`
- Destinations results: `Output/{research_area}_{beta}_Destinations_{timestamp}.csv`

### Log Files
- Training logs: `output_{research_area}_{prefix}_beta_{beta}.log`
- Performance logs: Various log files with timing information




---

**Note**: This project is designed for research purposes in maternal healthcare accessibility optimization. Ensure compliance with data privacy regulations when working with real healthcare data.

## Contact

For questions, suggestions, or collaborations, please contact: Yuhao Kang: yuhao.kang@austin.utexas.edu, Andy Qin: andy.qin@utexas.edu


