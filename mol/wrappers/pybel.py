from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
from openbabel import pybel

from .base import AbstractMol


class PybelMol(pybel.Molecule, AbstractMol):
    """
    Wrapper for pybel.Molecule.
    """
    def __init__(self, *args, smiles: str = None, **kwargs) -> None:
        """
        Initialize from SMILES string or from pybel.Molecule.

        Args:
            *args: Arguments for pybel.Molecule constructor if smiles is None, otherwise arguments for pybel.readstring.
            smiles: SMILES string.
            **kwargs: Keyword arguments for pybel.Molecule constructor if smiles is None, otherwise keywords arguments
                      for pybel.readstring.
        """
        if smiles is None:
            mol = args[0]
            if hasattr(mol, 'OBMol'):
                # Need to use OBMol in constructor so that title and properties are not lost
                mol = mol.OBMol
            super().__init__(mol, *args[1:], **kwargs)
        else:
            mol = pybel.readstring('smi', smiles, *args, **kwargs)
            super().__init__(mol)

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def name(self) -> str:
        return self.title

    @name.setter
    def name(self, val: str) -> None:
        self.title = val

    @property
    def smiles(self) -> str:
        """Canonical SMILES"""
        # 'n' doesn't include name in SMILES
        return self.to('can', n=None)

    @property
    def symbols(self) -> list[str]:
        """List of atomic symbols"""
        return [pybel.ob.GetSymbol(atom.atomicnum) for atom in self]

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates"""
        return np.array([atom.coords for atom in self])

    @coords.setter
    def coords(self, val: Union[np.ndarray, tuple[Sequence[str], np.ndarray]]) -> None:
        """
        Update coordinates and optionally check that atomic symbols match.

        Args:
            val: Numpy array of just the coordinates or tuple of symbols and coordinates.
        """
        symbols, coords = (None, val) if isinstance(val, np.ndarray) else val

        for i, (atom, xyz) in enumerate(zip(self, coords)):
            if symbols is not None:
                symbol = pybel.ob.GetSymbol(atom.atomicnum)
                if symbol != symbols[i]:
                    raise ValueError(f'Symbol {symbols[i]} does not match atom {atom.idx} of type {symbol}')

            atom.OBAtom.SetVector(*xyz)

    def copy(self) -> PybelMol:
        return self.clone

    def merge(self, other: PybelMol) -> PybelMol:
        """
        Combine self and other into new molecule.

        Args:
            other: Other PybelMol.

        Returns:
            Merged PybelMol.
        """
        obmol = self.copy().OBMol
        obmol += other.OBMol
        return type(self)(obmol)

    def get_prop(self, name: str) -> Any:
        """
        Get property stored in pybel.Molecule.data.

        Args:
            name: Name of property.

        Returns:
            Value of property.
        """
        return self.data[name]

    def set_prop(self, name: str, val: Any) -> None:
        """
        Set property.

        Args:
            name: Name of property.
            val: Value of property.
        """
        self.data[name] = val

    def to(self, frmt: str, **kwargs) -> str:
        """
        Convert to string representation.

        Args:
            frmt: Format.
            **kwargs: Options for pybel.Molecule.write.

        Returns:
            String representation in new format.
        """
        return self.write(frmt.lower(), opt=kwargs).rstrip()

    def save(self, path: Union[str, Path], frmt: str = None, **kwargs) -> None:
        """
        Write to file. Infer format from path extension if frmt is None.

        Args:
            path: File path.
            frmt: File format.
            **kwargs: Other keyword arguments for pybel.Molecule.write.
        """
        if frmt is None:
            path = Path(path)
            frmt = path.suffix[1:]

        overwrite = kwargs.pop('overwrite', False)  # Don't overwrite by default
        self.write(frmt.lower(), filename=str(path), overwrite=overwrite, opt=kwargs)

    def set_partial_charges(self, charges: Union[Iterable[float], np.ndarray], symbols: Sequence[str] = None) -> None:
        """
        Set atomic partial charges.

        Args:
            charges: Atomic partial charges.
            symbols: If provided, use these atomic symbols to verify against the existing ones.
        """
        self.OBMol.SetAutomaticPartialCharge(False)  # Need this to be able to set charges if none exist yet

        for i, (atom, charge) in enumerate(zip(self, charges)):
            if symbols is not None:
                symbol = pybel.ob.GetSymbol(atom.atomicnum)
                if symbol != symbols[i]:
                    raise ValueError(f'Symbol {symbols[i]} does not match atom {atom.idx} of type {symbol}')

            atom.OBAtom.SetPartialCharge(charge)

    def generate_3d_geometry(self, forcefield: str = 'mmff94', nconf: int = 1) -> None:
        """
        Generate conformer using force field minimization. Adds explicit
        hydrogens if they don't already exist.

        Args:
            forcefield: Force field type.
            nconf: Number of conformers to embed.
        """
        self.addh()
        for atom in self:
            atom.OBAtom.SetVector(0, 0, 0)

        self.make3D(forcefield=forcefield)
        if nconf > 1:
            self.find_lowest_energy_conformer(forcefield=forcefield, nconf=nconf)

    def optimize_geometry(self, forcefield: str = 'mmff94', steps: int = 500) -> None:
        """
        Perform a force field optimization for all conformers and set
        the lowest energy conformer.

        Args:
            forcefield: Force field type.
            steps: Number of optimization steps.
        """
        nconf = self.OBMol.NumConformers()

        if nconf:
            ff = pybel.ob.OBForceField.FindForceField(forcefield)

            if ff.Setup(self.OBMol):  # Returns True if successful
                min_energy = float('inf')
                cidx_min = None

                for cidx in range(nconf):
                    self.OBMol.SetConformer(cidx)
                    self.localopt(forcefield=forcefield, steps=steps)
                    energy = ff.Energy()

                    if energy < min_energy:
                        cidx_min = cidx
                        min_energy = energy

                if cidx_min is not None:
                    self.OBMol.SetConformer(cidx_min)
                else:
                    self.OBMol.SetConformer(0)
        else:
            self.localopt(forcefield=forcefield, steps=steps)

    def find_lowest_energy_conformer(self, forcefield: str = 'mmff94', nconf: int = 10, steps: int = 500) -> None:
        """
        Perform a weighted rotor conformer search to obtain the lowest energy conformer.

        Args:
            forcefield: Force field type.
            nconf: Number of conformers to consider.
            steps: Number of steps per conformer.
        """
        if not self.OBMol.Has3D():
            raise Exception(f'Molecule does not have 3D coordinates')

        ff = pybel.ob.OBForceField.FindForceField(forcefield)

        if ff.Setup(self.OBMol):  # Returns True if successful
            # initial_energy = ff.Energy()
            ff.WeightedRotorSearch(nconf, steps)
            # final_energy = ff.Energy()
        else:
            raise Exception('Force field could not be set up')

    def ga_conformer_search(self, nconf: int = 100) -> None:
        """
        Generate a diverse set of conformers using a genetic algorithm.
        It is recommended to call self.optimize_geometry afterwards.

        Args:
            nconf: Maximum number of conformers to generate.
        """
        if not self.OBMol.Has3D():
            raise Exception(f'Molecule does not have 3D coordinates')

        conf = pybel.ob.OBConformerSearch()
        conf.Setup(self.OBMol, nconf)
        conf.Search()
        conf.GetConformers(self.OBMol)
        # nconf_actual = self.OBMol.NumConformers()
