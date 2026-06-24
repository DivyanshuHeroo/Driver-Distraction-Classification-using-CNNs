"""Shared helpers: config loading and small filesystem utilities."""

import os
import yaml


def load_config(config_path="config.yaml"):
    """Load the YAML configuration file into a dictionary."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_class_names(config):
    """Return ordered list of class folder names (c0..c9)."""
    return sorted(config["classes"].keys())


def get_class_labels(config):
    """Return human-readable labels in c0..c9 order."""
    return [config["classes"][c] for c in get_class_names(config)]


def ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path
