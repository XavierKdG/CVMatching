## This is the Work In Progress (WIP) version of this project 


### Clone Repository
```bash
git clone https://github.com/XavierKdG/CVMatching.git
cd CVMatching
```

### Add Dataset
download the datasets below and place the it in the `data/raw/` directory

- [Resume.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=Resume.csv)

- [job_descriptions2.csv](https://www.kaggle.com/datasets/014ce6313a60bf1563ff9ef3d57879bd8e7c1e1be0e8926bffb82d51ee85fda8?select=job_descriptions2.csv)

### Install Conda Environment

1. Install Miniconda:

```bash
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash miniconda.sh
source ~/.bashrc
rm miniconda.sh
```

2. Create the Conda environment:

```
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate cvmatching
```
