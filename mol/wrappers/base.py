from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np


class AbstractMol(ABC):
    """
    Base class for pybel.Molecule and rdkit.Chem.Mol wrappers.
    """
    @abstractmethod
    def __init__(self, *args, smiles: str = None, **kwargs) -> None:
        """Should be able to instantiate subclass from SMILES"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of atoms"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @name.setter
    @abstractmethod
    def name(self, val: str) -> None:
        pass

    @property
    @abstractmethod
    def smiles(self) -> str:
        """Canonical SMILES"""
        pass

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """List of atom symbols"""
        pass

    @property
    @abstractmethod
    def coords(self) -> np.ndarray:
        """Cartesian coordinates"""

    @coords.setter
    @abstractmethod
    def coords(self, val: Union[np.ndarray, tuple[Sequence[str], np.ndarray]]) -> None:
        """Check atomic symbols if setting with tuple of symbols and coordinates"""
        pass

    @abstractmethod
    def copy(self) -> AbstractMol:
        pass

    @abstractmethod
    def merge(self, other: AbstractMol) -> AbstractMol:
        pass

    @abstractmethod
    def get_prop(self, name: str) -> Any:
        pass

    @abstractmethod
    def set_prop(self, name: str, val: Any) -> None:
        pass

    @abstractmethod
    def to(self, frmt: str) -> str:
        """Convert to string format"""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path], frmt: str) -> None:
        """Write to file"""
        pass

    @abstractmethod
    def generate_3d_geometry(self) -> None:
        pass

    @abstractmethod
    def optimize_geometry(self) -> None:
        pass
