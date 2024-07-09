"""This module provides the RAG-CTL config functionality"""
# ragctl/config.py

import configparser
from pathlib import Path
import typer
import os

from ragctl import (
    DB_WRITE_ERROR, DIR_ERROR, FILE_ERROR, SUCCESS, __app_name__, AWS_CONFIG_ERROR, AWS_KEY_ERROR
)
CONFIG_DIR_PATH = Path(typer.get_app_dir(__app_name__))
# Check if CONFIG DIR PATH is available
if not CONFIG_DIR_PATH.exists():
    CONFIG_DIR_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_FILE_PATH = CONFIG_DIR_PATH / "config.ini"

def init_app(db_path: str) -> int:
    """Initialize the application"""
    config_code = _init_config_file()
    if config_code != SUCCESS:
        return config_code
    database_code = _create_database(db_path)
    if database_code != SUCCESS:
        return config_code
    return SUCCESS

def init_aws(aws_access_key_id: str, aws_secret_access_key: str, aws_region: str) -> int:
    """
    Create an AWS credential configuration file with the provided access key ID and secret access key.

    :param aws_access_key_id: The AWS access key ID
    :param aws_secret_access_key: The AWS secret access key
    :return: None
    """
    try:
        # Define the AWS credential file path
        aws_credential_file_path = os.path.join(os.path.expanduser("~"), ".aws", "credentials")
        
        # Define the AWS config file path
        aws_config_file_path = os.path.join(os.path.expanduser("~"), ".aws", "config")

        # Create the credentials file if it doesn't exist
        if not os.path.exists(aws_credential_file_path):
            os.makedirs(os.path.dirname(aws_credential_file_path), exist_ok=True)
        # Check if aws_access_key_id and aws_secret_access_key are provided
        if not aws_access_key_id or not aws_secret_access_key:
            return AWS_KEY_ERROR
        
        # Write the credentials to the file
        with open(aws_credential_file_path, "w") as f:
            f.write(f"[default]\n")
            f.write(f"aws_access_key_id = {aws_access_key_id}\n")
            f.write(f"aws_secret_access_key = {aws_secret_access_key}\n")
        
        # Write the region to the config file
        with open(aws_config_file_path, "w") as f:
            f.write(f"[default]\n")
            f.write(f"region = {aws_region}\n")
            f.write(f"output = json\n")
        return SUCCESS
    except Exception as e:
        return AWS_CONFIG_ERROR

def _init_config_file() -> int:
    try:
        CONFIG_DIR_PATH.mkdir(exist_ok=True)
    except OSError:
        return DIR_ERROR
    try:
        CONFIG_FILE_PATH.touch(exist_ok=True)
    except OSError:
        return FILE_ERROR
    return SUCCESS

def _create_database(db_path: str) -> int:
    config_parser = configparser.ConfigParser()
    config_parser["General"] = {"database": db_path}
    try:
        with CONFIG_FILE_PATH.open("w") as file:
            config_parser.write(file)
    except OSError:
        return DB_WRITE_ERROR
    return SUCCESS