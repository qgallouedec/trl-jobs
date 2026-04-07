from importlib.resources import files

__version__ = files("trl_sagemaker").joinpath("version.txt").read_text().strip()
