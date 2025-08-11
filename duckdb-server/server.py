import logging
import sys
from mcp.server.fastmcp import FastMCP
from typing import Optional, Tuple
from dotenv import load_dotenv
import duckdb


# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("duckdb-mcp")

mcp = FastMCP("DuckDB-Server")

db_path = "~/ddb_agent4.db"

connection = duckdb.connect(db_path)

@mcp.prompt()
async def database_prompt() -> str:
    """
        Prompt to perform any database related operations in DubcDB
        - read data from database
        - write data into database
        - create table in database
        - load csv file
        - describe table from database
    """
    return [

        {"role": "system", "content": """
        you are a data agent that can perform all Data Engineering actions by 
        initialize the connection with DuckDB database Server and provides several DuckDB SQL related tools"""
        }, 
        {"role": "user", "content": """Please perform all below data actions in database:
            "run_query": Execute Select queries to read data from database
            "write_querty": Execute INSERT, UPDATE and DELETE queries to modify data
            "create_table": Creates new tables in database
            "list_tables": Show all existing tables
            "load_local_csv_to_table": load csv from local path as well as URL
            "describe_table": Shows the schema for a specific table
            "append_insight": Adds a new business insights to the recources"""
         }
    ]


@mcp.tool()
async def show_tables(show_tables: bool) -> str:
    """Function to show tables in the database

    :param show_tables: Show tables in the database
    :return: List of tables in the database
    """
    if show_tables:
        stmt = "SHOW TABLES;"
        tables = run_query(stmt)
        logger.debug(f"Tables: {tables}")
        return tables
    return "No tables to show"

@mcp.tool()
async def describe_table(table: str) -> str:
    """Function to describe a table

    :param table: Table to describe
    :return: Description of the table
    """
    stmt = f"DESCRIBE {table};"
    table_description = run_query(stmt)

    logger.debug(f"Table description: {table_description}")
    return f"{table}\n{table_description}"

@mcp.tool()
async def inspect_query(query: str) -> str:
    """Function to inspect a query and return the query plan. Always inspect your query before running them.

    :param query: Query to inspect
    :return: Query plan
    """
    stmt = f"explain {query};"
    explain_plan = run_query(stmt)

    logger.debug(f"Explain plan: {explain_plan}")
    return explain_plan

@mcp.tool()
async def run_query(query: str) -> str:
    """Function that runs a query and returns the result.

    :param query: SQL query to run
    :return: Result of the query
    """

    # -*- Format the SQL Query
    # Remove backticks
    formatted_sql = query.replace("`", "")
    # If there are multiple statements, only run the first one
    formatted_sql = formatted_sql.split(";")[0]

    try:
        logger.info(f"Running: {formatted_sql}")

        query_result = connection.sql(formatted_sql)
        result_output = "No output"
        if query_result is not None:
            try:
                results_as_python_objects = query_result.fetchall()
                result_rows = []
                for row in results_as_python_objects:
                    if len(row) == 1:
                        result_rows.append(str(row[0]))
                    else:
                        result_rows.append(",".join(str(x) for x in row))

                result_data = "\n".join(result_rows)
                result_output = ",".join(query_result.columns) + "\n" + result_data
            except AttributeError:
                result_output = str(query_result)

        logger.debug(f"Query result: {result_output}")
        return result_output
    except duckdb.ProgrammingError as e:
        return str(e)
    except duckdb.Error as e:
        return str(e)
    except Exception as e:
        return str(e)

@mcp.tool()
async def summarize_table(table: str) -> str:
    """Function to compute a number of aggregates over a table.
    The function launches a query that computes a number of aggregates over all columns,
    including min, max, avg, std and approx_unique.

    :param table: Table to summarize
    :return: Summary of the table
    """
    table_summary = run_query(f"SUMMARIZE {table};")

    logger.debug(f"Table description: {table_summary}")
    return table_summary

@mcp.tool()
async def get_table_name_from_path(path: str) -> str:
    """Get the table name from a path

    :param path: Path to get the table name from
    :return: Table name
    """
    import os

    # Get the file name from the path
    file_name = path.split("/")[-1]
    # Get the file name without extension from the path
    table, extension = os.path.splitext(file_name)
    # If the table isn't a valid SQL identifier, we'll need to use something else
    table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

    return table

@mcp.tool()
async def create_table_from_path(path: str, table: Optional[str] = None, replace: bool = False) -> str:
    """Creates a table from a path

    :param path: Path to load
    :param table: Optional table name to use
    :param replace: Whether to replace the table if it already exists
    :return: Table name created
    """

    if table is None:
        table = get_table_name_from_path(path)

    logger.debug(f"Creating table {table} from {path}")
    create_statement = "CREATE TABLE IF NOT EXISTS"
    if replace:
        create_statement = "CREATE OR REPLACE TABLE"

    create_statement += f" '{table}' AS SELECT * FROM '{path}';"
    run_query(create_statement)
    logger.debug(f"Created table {table} from {path}")
    return table

@mcp.tool()
async def export_table_to_path(table: str, format: Optional[str] = "PARQUET", path: Optional[str] = None) -> str:
    """Save a table in a desired format (default: parquet)
    If the path is provided, the table will be saved under that path.
        Eg: If path is /tmp, the table will be saved as /tmp/table.parquet
    Otherwise it will be saved in the current directory

    :param table: Table to export
    :param format: Format to export in (default: parquet)
    :param path: Path to export to
    :return: None
    """
    if format is None:
        format = "PARQUET"

    logger.debug(f"Exporting Table {table} as {format.upper()} to path {path}")
    if path is None:
        path = f"{table}.{format}"
    else:
        path = f"{path}/{table}.{format}"
    export_statement = f"COPY (SELECT * FROM {table}) TO '{path}' (FORMAT {format.upper()});"
    result = run_query(export_statement)
    logger.debug(f"Exported {table} to {path}/{table}")
    return result

@mcp.tool()
async def load_local_path_to_table(path: str, table: Optional[str] = None) -> Tuple[str, str]:
    """Load a local file into duckdb

    :param path: Path to load
    :param table: Optional table name to use
    :return: Table name, SQL statement used to load the file
    """
    import os

    logger.debug(f"Loading {path} into duckdb")

    if table is None:
        # Get the file name from the s3 path
        file_name = path.split("/")[-1]
        # Get the file name without extension from the s3 path
        table, extension = os.path.splitext(file_name)
        # If the table isn't a valid SQL identifier, we'll need to use something else
        table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

    create_statement = f"CREATE OR REPLACE TABLE '{table}' AS SELECT * FROM '{path}';"
    run_query(create_statement)

    logger.debug(f"Loaded {path} into duckdb as {table}")
    return table, create_statement

@mcp.tool()
async def load_local_csv_to_table(
    path: str, table: Optional[str] = None, delimiter: Optional[str] = None
) -> Tuple[str, str]:
    """Load a local CSV file into duckdb

    :param path: Path to load
    :param table: Optional table name to use
    :param delimiter: Optional delimiter to use
    :return: Table name, SQL statement used to load the file
    """
    import os

    logger.debug(f"Loading {path} into duckdb")

    if table is None:
        # Get the file name from the s3 path
        file_name = path.split("/")[-1]
        # Get the file name without extension from the s3 path
        table, extension = os.path.splitext(file_name)
        # If the table isn't a valid SQL identifier, we'll need to use something else
        table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

    select_statement = f"SELECT * FROM read_csv('{path}'"
    if delimiter is not None:
        select_statement += f", delim='{delimiter}')"
    else:
        select_statement += ")"

    create_statement = f"CREATE OR REPLACE TABLE '{table}' AS {select_statement};"
    run_query(create_statement)

    logger.debug(f"Loaded CSV {path} into duckdb as {table}")
    return table, create_statement

@mcp.tool()
async def load_s3_path_to_table(path: str, table: Optional[str] = None) -> Tuple[str, str]:
    """Load a file from S3 into duckdb

    :param path: S3 path to load
    :param table: Optional table name to use
    :return: Table name, SQL statement used to load the file
    """
    import os

    logger.debug(f"Loading {path} into duckdb")

    if table is None:
        # Get the file name from the s3 path
        file_name = path.split("/")[-1]
        # Get the file name without extension from the s3 path
        table, extension = os.path.splitext(file_name)
        # If the table isn't a valid SQL identifier, we'll need to use something else
        table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

    create_statement = f"CREATE OR REPLACE TABLE '{table}' AS SELECT * FROM '{path}';"
    run_query(create_statement)

    logger.debug(f"Loaded {path} into duckdb as {table}")
    return table, create_statement

@mcp.tool()
async def load_s3_csv_to_table(
    path: str, table: Optional[str] = None, delimiter: Optional[str] = None
) -> Tuple[str, str]:
    """Load a CSV file from S3 into duckdb

    :param path: S3 path to load
    :param table: Optional table name to use
    :return: Table name, SQL statement used to load the file
    """
    import os

    logger.debug(f"Loading {path} into duckdb")

    if table is None:
        # Get the file name from the s3 path
        file_name = path.split("/")[-1]
        # Get the file name without extension from the s3 path
        table, extension = os.path.splitext(file_name)
        # If the table isn't a valid SQL identifier, we'll need to use something else
        table = table.replace("-", "_").replace(".", "_").replace(" ", "_").replace("/", "_")

    select_statement = f"SELECT * FROM read_csv('{path}'"
    if delimiter is not None:
        select_statement += f", delim='{delimiter}')"
    else:
        select_statement += ")"

    create_statement = f"CREATE OR REPLACE TABLE '{table}' AS {select_statement};"
    run_query(create_statement)

    logger.debug(f"Loaded CSV {path} into duckdb as {table}")
    return table, create_statement

@mcp.tool()
async def create_fts_index(table: str, unique_key: str, input_values: list[str]) -> str:
    """Create a full text search index on a table

    :param table: Table to create the index on
    :param unique_key: Unique key to use
    :param input_values: Values to index
    :return: None
    """
    logger.debug(f"Creating FTS index on {table} for {input_values}")
    run_query("INSTALL fts;")
    logger.debug("Installed FTS extension")
    run_query("LOAD fts;")
    logger.debug("Loaded FTS extension")

    create_fts_index_statement = f"PRAGMA create_fts_index('{table}', '{unique_key}', '{input_values}');"
    logger.debug(f"Running {create_fts_index_statement}")
    result = run_query(create_fts_index_statement)
    logger.debug(f"Created FTS index on {table} for {input_values}")

    return result

@mcp.tool()
async def full_text_search(table: str, unique_key: str, search_text: str) -> str:
    """Full text Search in a table column for a specific text/keyword

    :param table: Table to search
    :param unique_key: Unique key to use
    :param search_text: Text to search
    :return: None
    """
    logger.debug(f"Running full_text_search for {search_text} in {table}")
    search_text_statement = f"""SELECT fts_main_corpus.match_bm25({unique_key}, '{search_text}') AS score,*
                                    FROM {table}
                                    WHERE score IS NOT NULL
                                    ORDER BY score;"""

    logger.debug(f"Running {search_text_statement}")
    result = run_query(search_text_statement)
    logger.debug(f"Search results for {search_text} in {table}")

    return result


if __name__ == "__main__":
    # Log server startup
    logger.info("Starting DuckDB MCP Server...")

    # Initialize and run the server
    mcp.run(transport="stdio")

    # This line won't be reached during normal operation
    logger.info("Server stopped")