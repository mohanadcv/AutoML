"""
MLflow Setup Module
Handles the creation of required directories and initializes MLflow tracking.
"""

import mlflow
import logging
from Config.config import Config

logger = logging.getLogger(__name__)

class MlflowSetup:

    def __init__(self, config: Config = Config(), experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name or self.config.MLFLOW_EXPERIMENT_NAME

    # --------------------------------------------------------
    def setup_mlflow(self):
        mlflow.set_tracking_uri(f"file:///{self.config.EXPERIMENT_DIR}")
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"✅ MLflow setup complete with experiment: {self.experiment_name}")
        logger.info(f"📁 MLflow tracking directory: {self.config.EXPERIMENT_DIR}")

    # --------------------------------------------------------
    def run_full_setup(self):
        self.setup_mlflow()
