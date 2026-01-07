"""Configuration management"""
import yaml
from pathlib import Path


class Config:
    """Configuration manager"""

    def __init__(self, config_path='config.yaml'):
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()

    @staticmethod
    def get_default_config():
        """Get default configuration"""
        return {
            'data': {
                'raw_dir': 'data/raw',
                'processed_dir': 'data/processed',
                'features_dir': 'data/features'
            },
            'models': {
                'output_dir': 'models',
                'random_state': 42
            },
            'results': {
                'output_dir': 'results',
                'figures_dir': 'results/figures',
                'tables_dir': 'results/tables'
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs'
            }
        }

    def __getitem__(self, key):
        return self.config[key]

    def get(self, key, default=None):
        return self.config.get(key, default)
