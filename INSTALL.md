# Installation Guide

## Recommended: Python 3.8–3.10

The project is tested with **Python 3.8**. Newer Python versions (3.12, 3.13) may cause build failures with some dependencies.

```bash
conda create -n kinesis python=3.10
conda activate kinesis
pip install -r requirements.txt
pip install torch  # or torch with CUDA for Linux
```

## If using Python 3.12+

Some packages (CLIP, chumpy, numpy 1.21) may fail. Try:

1. **Install setuptools first:**
   ```bash
   pip install setuptools pip --upgrade
   ```

2. **Use relaxed requirements** (drop strict version pins):
   ```bash
   pip install chumpy easydict gym gymnasium h5py hydra-core joblib lxml matplotlib mujoco numpy omegaconf pandas scipy torch wandb
   pip install git+https://github.com/ZhengyiLuo/smplx.git@a5b8e4ac14f79f3f33fd2cf2a16e6f507146b813
   ```

3. **CLIP** (for text-to-motion) is commented out in `requirements.txt` because it fails on Python 3.12+. Install separately if needed:
   ```bash
   pip install git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
   ```

## macOS

```bash
conda create -n kinesis python=3.10
conda activate kinesis
pip install -r requirements.txt
conda install -c conda-forge lxml
```
