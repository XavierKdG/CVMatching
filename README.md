## CV-Matching Project


### Clone Repository
```bash
git clone https://github.com/XavierKdG/CVMatching.git
cd CVMatching
```

### Add Dataset
Download de datasets hieronder en plaats ze in de map `data/raw/`:

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
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate cvmatching
```

### Project Structure
- **notebooks/**: Bevat notebooks waarin keuzes, methodes en evaluaties worden uitgelegd. Hier vind je toelichting waarom bepaalde modellen, evaluaties en preprocessing-stappen zijn gekozen.
- **src/**: Bevat de hoofdcode om het project uit te voeren:
  - `preprocess.py`: scripts voor het voorbereiden van CV- en vacaturedata
  - `finetune.py`: scripts voor het trainen van TSDAE of andere embeddings
  - `evaluate.py`: scripts voor het evalueren van de modellen op mutual ranking, hubness, perturbation en andere metrics
- `data`: bevat de raw en processed datasets
- `results`: bevat de resultaten van de vergelijking van de modellen in json formaat
- `models`: bevat de modellen die getrained zijn en worden opgeslagen.

Om het project correct uit te voeren, volg deze volgorde:
```bash
1. preprocess -> 2. tsdae_fine_tuning -> 3. evaluate
```

### Usage
Na setup kun je de scripts uit `src/` uitvoeren om data te preprocessen, modellen te trainen en te evalueren. De notebooks bieden aanvullende uitleg vam resultaten en keuzes.



