import logging

logger = logging.getLogger("MLApp")
logger.setLevel(logging.INFO)

# Консольный вывод
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Формат
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)

# Добавляем хендлер к логгеру
if not logger.hasHandlers():
    logger.addHandler(console_handler)
