import psycopg2
from psycopg2.extras import RealDictCursor
from config.configure import CONNECTION_STRING
from utils.helpers import setup_logger
from utils.helpers import red, green, reset

logger = setup_logger(__name__)


def get_db_connection():
    try:
        if not CONNECTION_STRING:
            raise ValueError("DB_CONNECTION not found in environment variables.")

        connection = psycopg2.connect(CONNECTION_STRING, cursor_factory=RealDictCursor)
        # logger.info(green + "Database connection established successfully." + reset)
        return connection

    except psycopg2.OperationalError as db_error:
        logger.error(red + f"Database operational error: {db_error}" + reset)
        raise

    except Exception as e:
        logger.error(red + f"Unexpected database connection error: {e}" + reset)
        raise
