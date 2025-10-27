from .preprocess import main as preprocess_data
from .train import main as train_model
from .upload_to_qdrant import main as upload_to_qdrant
from .utils import setup_logging, load_config