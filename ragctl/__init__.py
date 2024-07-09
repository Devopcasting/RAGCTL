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
    DOC_DUPLICATE_ERROR,
    DOC_ALREADY_EMBEDDED,
    INVALID_PDF_FILE,
    AWS_CONFIG_ERROR,
    AWS_KEY_ERROR,
    EMBEDDING_ERROR
) = range(14)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "document id not found",
    DOC_PATH_ERROR: "documents path not found",
    DOC_DUPLICATE_ERROR: "duplicate document",
    DOC_ALREADY_EMBEDDED: "document already embedded",
    INVALID_PDF_FILE: "invalid pdf file",
    AWS_CONFIG_ERROR: "aws configuration error",
    AWS_KEY_ERROR: "aws key error",
    EMBEDDING_ERROR: "embedding error"
}