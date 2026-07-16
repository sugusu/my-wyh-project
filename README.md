# my-wyh-project

Code repository for research experiments: **DeformTransGS** and **RecycleGS**.

## Contents

- `DeformTransGS/` - Deformable 3D Gaussian Splatting experiments
- `RecycleGS/` - Recycling 3D Gaussian Splatting experiments

## Setup

```bash
# 1. Clone code
git clone https://github.com/sugusu/my-wyh-project.git
cd my-wyh-project

# 2. Download data from HuggingFace
# Install: pip install huggingface-hub
# Login:  hf auth login
./download_data.sh
```

## Data

Large files hosted on HuggingFace:

| Repo | Contents | Size |
|------|----------|------|
| [sugusu/my-pro](https://huggingface.co/sugusu/my-pro) | RecycleGS data/baselines/outputs + DeformTransGS GT experiments | ~15 GB |
| [sugusu/my-pro-data](https://huggingface.co/datasets/sugusu/my-pro-data) | DeformTransGS stage4_0_attribute_sufficiency_gate | ~1.3 GB |

Run `./download_data.sh` after cloning to fetch all data.
