from pathlib import Path

class Config:
    # REPRODUCIBILITY SETTINGS
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    CV_FOLDS = 5
    N_JOBS = -1


    # LOGGING
    LOG_LEVEL: str = "INFO"


    # PATHS
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_DIR = BASE_DIR / "models"
    EXPERIMENT_DIR = BASE_DIR / "experiments"


    # MLFLOW SETTINGS
    MLFLOW_TRACKING_URI = f"file:///{EXPERIMENT_DIR}"
    MLFLOW_EXPERIMENT_NAME = "automl_experiments"


    # FILE HANDLING LIMITS (SAFETY!)
    MAX_FILE_SIZE_MB = 100  # Maximum upload size
    ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']  # Supported file types
    MIN_ROWS = 10  # Minimum rows required
    MIN_COLUMNS = 2  # Minimum columns required
    MAX_NULL_RATIO = 0.5  # Maximum allowed missing data ratio


    # TASK DETECTION SETTINGS
    # Threshold for classification vs regression
    UNIQUE_RATIO_THRESHOLD = 0.05  # If unique_values/total_rows < 0.05 → classification
    MAX_UNIQUE_FOR_CLASSIFICATION = 20  # If unique_values < 20 → likely classification


    # Default models to include
    DEFAULT_CLASSIFICATION_MODELS = [
        'Logistic Regression',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost',
        'LightGBM'
    ]

    DEFAULT_REGRESSION_MODELS = [
        'Linear Regression',
        'Ridge',
        'Random Forest',
        'Gradient Boosting',
        'XGBoost',
        'LightGBM'
    ]


    #Create directories if they don't exist
    @classmethod
    def setup_directories(cls):
        """Create all required directories"""
        directories = [
            cls.MODEL_DIR,
            cls.EXPERIMENT_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)