# K-Sim Zbot Deployment 


## Setup


## Getting Started
1. Set up Git LFS and pull large files

```bash
# Install Git LFS
git lfs install

# Pull large files (URDF models, neural networks, etc.)
git lfs pull
```

2. Clone the repository

```bash
git clone git@github.com:kscalelabs/ksim-zbot.git
```

3. Make sure you're using Python 3.11 or greater

```bash
python3.11 --version  # Should show Python 3.11 or greater
```

4. Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

5. Initalize submodules
```bash
git submodule update --init --recursive
```

6. Runing
```bash
python -m ksim_zbot.zbot2.deploy.deploy_sim --model_path ksim_zbot/zbot2/walking/zbot_walking_task/run_13/checkpoints/tf_model --episode_length 20
```