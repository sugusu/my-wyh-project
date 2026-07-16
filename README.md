# my-wyh-project

Code repository for research experiments.

## Contents

- **DeformTransGS/** - Deformable 3D Gaussian Splatting experiments
- **RecycleGS/** - Recycling 3D Gaussian Splatting experiments

## Setup

```bash
# 1. Clone code
git clone https://github.com/sugusu/my-wyh-project.git
cd my-wyh-project

# 2. Download data from HuggingFace
# Install hf CLI: pip install huggingface-hub
# Login: hf auth login
./download_data.sh
```

## Data

Large files (experiments, datasets, model checkpoints) are hosted on HuggingFace:
https://huggingface.co/sugusu/my-pro

Run `./download_data.sh` after cloning to fetch them.
