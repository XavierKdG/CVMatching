import os
import argparse
import yaml
from sentence_transformers import SentenceTransformer

class SentenceTransformerTrainer:
    def __init__(self, config):
        self.config = config
        os.makedirs(config['model_dir'], exist_ok=True)
        self.model_name = config.get('pretrained_model', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(self.model_name)

    def save_model(self):
        save_path = os.path.join(self.config['model_dir'], self.config['model_name'])
        self.model.save(save_path)
        print(f"Pre-trained model saved at {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Save or fine-tune SentenceTransformer for CV-Job matching")
    parser.add_argument('--config', type=str, default='configs/train_yml.cfg', help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = SentenceTransformerTrainer(config)
    trainer.save_model()

if __name__ == "__main__":
    main()
