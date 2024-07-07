"""This module provides the RAG-CTL database functionality"""
# ragctl/model.py

import configparser
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple
from ragctl import DB_WRITE_ERROR, SUCCESS, DB_READ_ERROR, JSON_ERROR

DEFAULT_DB_FILE_PATH = Path.home().joinpath(
    "." + Path.home().stem + "_ragctl.json"
)

def get_database_path(config_file: Path) -> Path:
    """Return the current path to the ragctl database"""
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    return Path(config_parser["General"]["database"])

def init_database(db_path: Path) -> int:
    """Create the ragctl database"""
    try:
        db_path.write_text("[]")
        return SUCCESS
    except OSError:
        return DB_WRITE_ERROR

class DBResponse(NamedTuple):
    ragdoc_list: List[Dict[str, Any]]
    error: int

class DatabaseHandler:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def read_ragdocs(self) -> DBResponse:
        try:
            with open(self._db_path, "r") as db:
                try:
                    return DBResponse(json.load(db), SUCCESS)
                except json.JSONDecodeError:
                    return DBResponse([], JSON_ERROR)
        except OSError:
            return DBResponse([], DB_READ_ERROR)
    
    def write_ragdocs(self, ragdoc_list: List[Dict[str, Any]]) -> DBResponse:
        try:
            with open(self._db_path, "w") as db:
                json.dump(ragdoc_list, db, indent=4)
            return DBResponse(ragdoc_list, SUCCESS)
        except OSError:
            return DBResponse(ragdoc_list, DB_WRITE_ERROR)