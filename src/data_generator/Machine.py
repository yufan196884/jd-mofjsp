"""This file provides the Task class."""
# Standard package import
from typing import List


class Machine:
    """
    This class can be used to model tasks of a scheduling problem.
    Multiple tasks can be used to model jobs of a scheduling problem.

    :param standby_energy: consumption of machine when idle (KwH)
    """

    def __init__(self, standby_energy: float):
        self.standby_energy = standby_energy

    def __str__(self) -> str:
        return f"Machine(standby_energy={self.standby_energy})"

    def str_info(self) -> str:
        return f"Machine standby energy: {self.standby_energy} KwH"