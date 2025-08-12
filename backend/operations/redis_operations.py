import json
import redis
from typing import Any, Optional, List
from connections.redis_connect import get_redis_client
from utils.helpers import green, red, reset, setup_logger

logger = setup_logger(__name__)

redis_client: redis.Redis = get_redis_client()  # type: ignore


def set_cache(key: str, value: Any, expire: Optional[int] = None) -> bool:
    try:
        serialized_value = json.dumps(value)
        if expire is not None:
            redis_client.setex(key, expire, serialized_value)
        else:
            redis_client.set(key, serialized_value)
        # logger.info(green + f"Successfully cached key: '{key}'" + reset)
        return True
    except (TypeError, ValueError) as serialization_error:
        logger.error(
            red
            + f"Serialization failed for key: '{key}', Error: {serialization_error}"
            + reset
        )
        return False
    except redis.RedisError as redis_error:
        logger.error(
            red + f"Redis error for key: '{key}', Error: {redis_error}" + reset
        )
        return False
    except Exception as unexpected_error:
        logger.error(
            red
            + f"Unexpected error for key: '{key}', Error: {unexpected_error}"
            + reset
        )
        return False


def get_cache(key: str) -> Optional[Any]:
    try:
        cached_value = redis_client.get(key)
        if cached_value is None:
            # logger.info(blue + f"Cache miss for key: '{key}'" + reset)
            return None
        # logger.info(blue + f"Cache hit for key: '{key}'" + reset)
        return json.loads(cached_value)  # type: ignore[arg-type]
    except json.JSONDecodeError as json_error:
        logger.error(
            red + f"Failed to decode JSON for key: '{key}'. Error: {json_error}" + reset
        )
        return None
    except redis.RedisError as redis_error:
        logger.error(
            red + f"Redis error getting key: '{key}'. Error: {redis_error}" + reset
        )
        return None
    except Exception as e:
        logger.error(
            red + f"Unexpected error retrieving key: '{key}'. Error: {e}" + reset
        )
        return None


def get_keys(pattern: str) -> List[str]:
    try:
        keys = redis_client.keys(pattern)  # type: ignore
        # if keys:
        #     logger.info(
        #         blue + f"Found {len(keys)} keys matching pattern '{pattern}'." + reset  # type: ignore
        #     )
        # else:
        #     logger.info(blue + f"No keys found for pattern '{pattern}'." + reset)
        return keys  # type: ignore
    except redis.RedisError as redis_error:
        logger.error(
            red
            + f"Redis error fetching keys with pattern '{pattern}': {redis_error}"
            + reset
        )
        return []
    except Exception as e:
        logger.error(
            red + f"Unexpected error fetching keys for pattern '{pattern}': {e}" + reset
        )
        return []


def delete_cache(key: str) -> bool:
    try:
        result = redis_client.delete(key)
        if result == 1:
            # logger.info(green + f"Successfully deleted key: '{key}'" + reset)
            return True
        else:
            # logger.info(blue + f"Key '{key}' did not exist in Redis." + reset)
            return False
    except redis.RedisError as redis_error:
        logger.error(red + f"Redis error deleting key '{key}': {redis_error}" + reset)
        return False
    except Exception as e:
        logger.error(red + f"Unexpected error deleting key '{key}': {e}" + reset)
        return False


def persist_redis_key(key: str) -> bool:
    try:
        result = redis_client.persist(key)
        if result == 1:
            logger.info(green + f"Made Redis key persistent: '{key}'" + reset)
            return True
        else:
            # logger.info(
            #     blue + f"Key '{key}' was already persistent or does not exist." + reset
            # )
            return False
    except redis.RedisError as redis_error:
        logger.error(
            red + f"Redis error making key '{key}' persistent: {redis_error}" + reset
        )
        return False
    except Exception as e:
        logger.error(red + f"Unexpected error persisting key '{key}': {e}" + reset)
        return False
