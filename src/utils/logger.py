import logging
from src.utils.logger_interface import Ilogger

class Logger(Ilogger):
    def __init__(self):
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)  # Captura todos los logs

        # Verificar si los handlers ya fueron añadidos
        if not self.logger.hasHandlers():
            self.formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Handler para archivo
            self.file_handler = logging.FileHandler("log.log", encoding="utf-8")
            self.file_handler.setLevel(logging.DEBUG)
            self.file_handler.setFormatter(self.formatter)

            # Handler para consola
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(logging.INFO)
            self.console_handler.setFormatter(self.formatter)

            # Agregar handlers al logger
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)

    def log(self, message: str, level: str = "debug"):
        """
        Registra un mensaje en el log.
        :param message: Texto del log.
        :param level: Nivel del log ("debug", "info", "warning", "error", "critical").
        """
        level = level.lower()
        if level == "debug":
            self.logger.info(message)
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            self.logger.debug(
                message
            )  # Si no reconoce el nivel, lo registra como debug


class LoggerFactory:
    """
    Logger Factory, you can extend this class to create a new logger
    """

    @staticmethod
    def create_logger() -> Ilogger:
        """Devuelve una instancia del logger"""
        return Logger()
