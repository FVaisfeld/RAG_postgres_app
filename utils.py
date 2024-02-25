import yaml
import os
from dotenv import load_dotenv, find_dotenv



class Config:
    """Configuration management from a YAML file."""
    def __init__(self, filepath):
        self.configs = self._load_configs(filepath)
        self.dbname = self.configs['dbname']
        self.user = self.configs['user']
        self.json_path = self.configs['json_path']
        self.n_context = self.configs['n_context']
        self.dist_metric = self.configs['dist_metric']
        self.model = self.configs['model']
        self.temperature = self.configs['temperature']

    @staticmethod
    def _load_configs(filepath):
        """Load configuration from a YAML file."""
        try:
            with open(filepath, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file {filepath} not found.")
            return {}

        

def get_openai_api_key():
    """Retrieve OpenAI API key from environment variables."""
    _ = load_dotenv(find_dotenv())
    return os.getenv('openaiAPI')