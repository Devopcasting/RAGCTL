"""Top-level package for RAG-CTL"""
# ragctl/__init__.py

__app_name__ = "ragctl"
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
    DOC_PATH_ERROR,
    DOC_DUPLICATE_ERROR
) = range(9)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "document id error",
    DOC_PATH_ERROR: "documents path not found",
    DOC_DUPLICATE_ERROR: "duplicate document"
}