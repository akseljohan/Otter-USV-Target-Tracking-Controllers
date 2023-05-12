import yaml
from pathlib import Path

path = Path(__file__).parent

config = yaml.safe_load(open((path / 'config.yml')))

__all__ = [config]