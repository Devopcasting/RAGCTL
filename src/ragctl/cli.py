"""This module provides the RAG-CTL CLI"""
# ragctl/cli.py

from typing import Optional, List
from pathlib import Path
import typer
import re
from ragctl import (
    __app_name__, __version__, ERRORS, config, model, ragctl
)
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Retrieval Augmented Generation (RAG) AI CLI")

# Command: initialize the database
@app.command()
def init(
    db_path: str = typer.Option(
        str(model.DEFAULT_DB_FILE_PATH),
        "--db-path",
        "--db",
        prompt="ragctl database location?",
    ),
) -> None:
    """Initialize the ragctl database"""
    app_init_error = config.init_app(db_path)
    if app_init_error:
        typer.secho(
            f'Creating config file failed with "{ERRORS[app_init_error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    db_init_error = model.init_database(Path(db_path))
    if db_init_error:
        typer.secho(
            f'Creating database failed with "{ERRORS[db_init_error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    else:
        typer.secho(f"The RAGCTL database is {db_path}", fg=typer.colors.GREEN)

def validate_aws_access_key_id(value: str):
    if len(value) != 20 or not re.match(r'^[A-Za-z0-9]+$', value):
        raise typer.BadParameter("AWS Access Key ID must be 20 alphanumeric characters long.")
    return value

def validate_aws_secret_access_key(value: str):
    if len(value) != 40 or not re.match(r'^[A-Za-z0-9]+$', value):
        raise typer.BadParameter("AWS Secret Access Key must be 40 alphanumeric characters long.")
    return value

# Command: initialize aws credentials
@app.command()
def aws_config(
    aws_access_key_id: str = typer.Option(
        str(), "--aws-access-key-id", prompt="AWS Access Key ID", help="AWS Access Key ID", callback=validate_aws_access_key_id
    ),
    aws_secret_access_key: str = typer.Option(
        str(), "--aws-secret-access-key", prompt="AWS Secret Access Key", help="AWS Secret Access Key", hide_input=True, callback=validate_aws_secret_access_key
    ),
    aws_region: str = typer.Option(
        "us-east-1","--aws-region", prompt="AWS Region", help="AWS Region", show_default=True
    )
) -> None:
    """Configure aws credentials"""
    aws_init_error = config.init_aws(
        aws_access_key_id, aws_secret_access_key, aws_region
    )
    if aws_init_error:
        typer.secho(
            f'Creating aws credentials failed with "{ERRORS[aws_init_error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            "AWS credentials initialized successfully",
            fg=typer.colors.GREEN
        )

def get_ragdocs() -> ragctl.RagDocer:
    if config.CONFIG_FILE_PATH.exists():
        db_path = model.get_database_path(config.CONFIG_FILE_PATH)
    else:
        typer.secho(
            'Config file not found, Please run "ragctl init"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    if db_path.exists():
        return ragctl.RagDocer(db_path)
    else:
        typer.secho(
            'Database not found. Please, run "ragctl init"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    
# Command: Upload the list of documents
@app.command()
def upload(documents_path: List[str] = typer.Argument(..., help="Path to the PDF documents")) -> None:
    """Upload PDF documents"""
    ragdocer = get_ragdocs()
    ragdocer, error = ragdocer.upload_doc(documents_path)
    if error:
        typer.secho(
            f'Uploading documents failed with "{ERRORS[error]}"', fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        for result in ragdocer:
            if result["status"] == 1:
                typer.secho(
                    f"""ragctl: "{result['doc_path']}" was uploaded successfully""",
                    fg=typer.colors.GREEN
                )
            else:
                typer.secho(
                    f'ragctl: "{result["doc_path"]}" failed with "{result["message"]}"',
                    fg=typer.colors.RED
                )

# Command: Perform embeddings on the document id
@app.command(name="embedding")
def embed(
    doc_id: int = typer.Argument(..., help="ID of the document to embed")
) -> None:
    """Perform embeddings on the document id"""
    ragdocer = get_ragdocs()
    ragdocer, error = ragdocer.embed_document(doc_id)
    if error:
        typer.secho(
            f'Embedding document failed with "{ERRORS[error]}"',
            fg=typer.colors.RED
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            f"""ragctl: "{ragdocer['name']}" was embedded successfully""",
            fg=typer.colors.GREEN
        )
        
# Command: List all the uploaded documents
@app.command(name="list")
def list_all() -> None:
    """List all the uploaded documents"""
    ragdocer = get_ragdocs()
    documents = ragdocer.get_documents_list()
    if len(documents) == 0:
        typer.secho(
            'There are no documents in the database yet', fg=typer.colors.RED
        )
        raise typer.Exit()
    table = Table(title_justify="left")
    table.add_column("ID", style="bold", width=6)
    table.add_column("Name", width=40)
    table.add_column("Size", width=10)
    table.add_column("Embedding", width=9)
    for doc in documents:
        table.add_row(str(doc["id"]), doc["name"], doc["size"], doc["embedding"])
    # Display the table
    console = Console()
    console.print(table)
    typer.secho(
        f'Total uploaded documents: {len(documents)}', fg=typer.colors.GREEN
    )

# Command: List only those documents which are not embedded
@app.command(name="list-non-embedded")
def list_non_embedded() -> None:
    """List all the documents which are not embedded"""
    ragdocer = get_ragdocs()
    documents = ragdocer.get_non_embedded_documents()
    if len(documents) == 0:
        typer.secho(
            'There are no documents in the database yet', fg=typer.colors.RED
        )
        raise typer.Exit()
    table = Table(
        title="RAG-CTL: All non-embedded documents", title_justify="left"
    )
    table.add_column("ID", style="bold", width=6)
    table.add_column("Name", width=40)
    table.add_column("Size", width=10)
    table.add_column("Embedded", width=9)
    for doc in documents:
        table.add_row(str(doc["id"]), doc["name"], doc["size"], doc["embedding"])
    # Display the table
    console = Console()
    console.print(table)
    typer.secho(
        f'Total non-embedded documents: {len(documents)}',
        fg=typer.colors.GREEN
    )

# Command: List only those documents which are embedded
@app.command(name="list-embedded")
def list_embedded() -> None:
    """List all the documents which are embedded"""
    ragdocer = get_ragdocs()
    documents = ragdocer.get_embedded_documents()
    if len(documents) == 0:
        typer.secho(
            'There are no documents in the database yet', fg=typer.colors.RED
        )
        raise typer.Exit()
    table = Table(
        title="RAG-CTL: All embedded documents", title_justify="left"
    )
    table.add_column("ID", style="bold", width=6)
    table.add_column("Name", width=40)
    table.add_column("Size", width=10)
    table.add_column("Embedded", width=9)
    for doc in documents:
        table.add_row(str(doc["id"]), doc["name"], doc["size"], doc["embedded"])
    # Display the table
    console = Console()
    console.print(table)
    typer.secho(
        f'Total embedded documents: {len(documents)}',
        fg=typer.colors.GREEN
    )

# Command: Clear all the database
@app.command(name="clear")
def remove_all(
    force: bool = typer.Option(
        ...,
        prompt="Delete all the uploaded documents?",
        help="Force deletion without confirmation.",
    ),
) -> None:
    """Clear all the documents from the database"""
    ragdocer = get_ragdocs()
    if force:
        error = ragdocer.clear_all().error
        if error:
            typer.secho(
                f'Clearing documents failed with "{ERRORS[error]}"',
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)
        else:
            typer.secho(
                "All documents have been deleted", fg=typer.colors.GREEN
            )
    else:
        typer.echo("Operation canceled")

# Command: Delete a particular document
@app.command(name="delete")
def remove(
    doc_id: int = typer.Argument(..., help="ID of the document to delete")
) -> None:
    """Delete a particular document from the database"""
    ragdocer = get_ragdocs()
    error = ragdocer.delete_document(doc_id).error
    if error:
        typer.secho(
            f'Deleting document failed with "{ERRORS[error]}"',
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    else:
        typer.secho(
            f'ragctl: Document with ID "{doc_id}" has been deleted',
            fg=typer.colors.GREEN
        )

# Command: Query PDF document
@app.command(name="query")
def query(
    query: str = typer.Argument(..., help="Query to search for in the documents")
) -> None:
    """Ask questions to embedded PDF document"""
    ragdocer = get_ragdocs()
    results = ragdocer.query_documents(query)
    if not results:
        typer.secho(
            'No matching documents found', fg=typer.colors.RED
        )
        raise typer.Exit()
    typer.secho(
        results, fg=typer.colors.BRIGHT_WHITE
    )

def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__} (Chat with PDF)")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True
    )
) -> None:
    return