import redis
from config.configure import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
from utils.helpers import red, reset, green, setup_logger

logger = setup_logger(__name__)


_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """
    Initializes a Redis client if one doesn't exist,
    then returns the single client instance.
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD or None,
            db=REDIS_DB,
            decode_responses=True,
        )
        if client.ping():
            _redis_client = client
            return _redis_client
        else:
            logger.error(red + "Redis ping failed." + reset)
            return None
    except redis.ConnectionError as e:
        logger.error(red + f"Failed to connect to Redis: {e}" + reset)
        return None
