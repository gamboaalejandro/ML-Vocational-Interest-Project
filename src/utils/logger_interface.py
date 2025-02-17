from abc import ABC, abstractmethod


class Ilogger(ABC):
    """
    Interface for logger: log messages with different levels
    this interface goal the DIP
    """

    @abstractmethod
    def log(self, message: str, level: str) -> None:
        pass
