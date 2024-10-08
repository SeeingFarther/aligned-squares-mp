# Multi-Robot Motion Planning

This repository contains the implementation of a multi-robot motion planning algorithm that uses various sampling strategies to generate efficient motion paths. The implemented samplers include a Pair Sampler specifically designed for composite square robots.

### Prerequisites

- Python 3.10.9
  - Required Python packages (listed in `requirements.txt` or `conda_requirements.txt`):

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/multi-robot-motion-planning.git
    cd multi-robot-motion-planning
    ```

2. **Install the required packages:**

    ```bash
    pip3 install -r requirements.txt
    ```
   or
    ```bash
     conda create --name <env> --file <this file>
    ```

### Running the Experiments

You can run the experiments using the provided Python script. Below are the descriptions of the command-line arguments and examples of how to use them.

#### Command-Line Arguments

- `--compare-landmarks` (type: bool, default: False): Conduct landmarks hyperparameter experiments.
- `--compare-algo` (type: bool, default: False): Conduct algorithms experiments.
- `--compare-k` (type: bool, default: False): Conduct k hyperparameter experiments.
- `--compare-length` (type: bool, default: False): Conduct metric length experiments.
- `--k` (type: int, default: 15): Value of k.
- `--num-landmarks` (type: int, default: 1000): Number of landmarks.
- `--prm-num-landmarks` (type: int, default: 2000): Number of landmarks for PRM for DRRT.
- `--num_experiments` (type: int, default: 5): Number of experiments.
- `--bound` (type: int, default: 2): Bounding width factor.
- `--eps` (type: float, default: 5): Epsilon value(Staggred Grid).
- `--delta` (type: float, default: 2): Clearance(Staggred Grid).
- `--solver` (type: str, default: "squares", choices: ['prm', 'drrt', 'staggered', 'squares']): Type of solver.
- `--nearest_neighbors` (type: str, default: None, choices: ['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2', 'Mix_CTD','Mix_Epsilon_2']): Type of nearest neighbor metric.
- `--roadmap_nearest_neighbors` (type: str, default: None, choices: ['CTD', 'Euclidean', 'Epsilon_2', 'Epsilon_Inf', 'Max_L2', 'Mix_CTD','Mix_Epsilon_2']): Type of roadmap nearest neighbor metric.
- `--exact` (type: bool, default: False): Run exact number of successful experiments.
- `--path` (type: str, default: './scenes/easy2.json'): Path to scene file, not used with compare flags.
- `--sampler` (type: str, default: None, choices: ['uniform', 'combined']): Type of sampler.
- `--to-file` (type: bool, default: False): Write output to file.
- `--file` (type: str, default: './results/other_benchmarks_tests.txt'): Path to output file.
- `--append-to-file` (type: bool, default: False): Append the output to existing file.
- `--scene-dir` (type: str, default: './scenes/'): Path to scene directory used with compare flags.
- `--time_limit` (type: int, default: 200): Time limit in seconds for exact flag so we dont get stuck.

#### Example Usage

Here is an example of how to run the script with specific parameters(Running PRM with 500 landmarks on easy2 scene and uniform sampler)

```bash
python experiments.py  --num-landmarks 500 --solver prm --path ./scenes/Easy2.json --sampler uniform
```


### GUI

You can run squares_planner solver by loading it to the discopygal GUI(This recommended way to run the squares planner solver)

https://www.cs.tau.ac.il/~cgl/discopygal/docs/tutorials/using_discopygal.html

### Bash Script

Below is an example bash script to run experiments with predefined parameters:

```bash
#!/bin/bash

# Run experiments with predefined parameters
python experiments.py \
    --k 20 \
    --num-landmarks 1000 \
    --prm-num-landmarks 2000 \
    --num_experiments 10 \
    --bound 3 \
    --eps 10 \
    --delta 3 \
    --solver squares \
    --nearest_neighbors Euclidean \
    --roadmap_nearest_neighbors Euclidean \
    --exact True \
    --path ./scenes/complex_scene.json \
    --sampler combined \
    --to-file True \
    --file ./results/complex_scene_results.txt \
    --append-to-file True \
    --scene-dir ./scenes/ \
    --time_limit 300
```

Save the script above as `run_experiments.sh` and make it executable:

###### Note: That the script is running the squares solver on the complex_scene with combined sampler and writing the results to the file complex_scene_results.txt and some of the parameters are predefined but will not as they needed for other solvers it is just for the example.

```bash
chmod +x run_experiments.sh
```

Run the script:

```bash
./run_experiments.sh
```

This setup allows for flexible and repeatable experiments to test and evaluate the performance of different motion planning strategies in various environments.