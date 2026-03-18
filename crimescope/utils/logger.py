import sys
from loguru import logger

logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <white>{message}</white>",
    level="DEBUG",
    colorize=True,
)

logger.add(
    "logs/crimescope.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    level="INFO",
    rotation="10 MB",   
    retention="7 days",
    compression="zip",
)