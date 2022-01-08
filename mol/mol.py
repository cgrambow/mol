import functools
from pathlib import Path
from typing import Union

from openbabel import pybel
from rdkit import Chem

from .conversion import file_to_pybel, file_to_rdkit, pybel_to_rdkit, rdkit_to_pybel
from .wrappers import PybelMol, RDKitMol


def process_mol_type(mol_type: str) -> str:
    """
    Convert potential mol_type strings into valid mol_type strings.

    Args:
        mol_type: Unprocessed mol type.

    Returns:
        'pybel' or 'rdkit'.
    """
    mol_type_new = mol_type.lower()
    if mol_type_new in {'pybel', 'babel', 'obabel', 'openbabel'}:
        return 'pybel'
    elif mol_type_new == 'rdkit':
        return 'rdkit'
    else:
        raise ValueError(f'Invalid molecule type "{mol_type}"')


class Mol:
    """
    Wrapper for pybel.Molecule/rdkit.Chem.Mol.
    Change mol_type to change wrapper.
    """

    class _MolDecorators:
        """
        Example:
            @_MolDecorators.require('rdkit')
            def example_method_of_Mol(self, *args, **kwargs):
                pass
        """
        @staticmethod
        def require(mol_type):
            """Decorator to ensure that a specific underlying mol class is used"""
            def decorator(method):
                @functools.wraps(method)
                def wrapper(self, *args, **kwargs):
                    # Only trigger mol setter if mol type changed
                    if self.mol_type != mol_type:
                        msg = 'Converting mol '
                        if self.name:
                            msg += f'"{self.name}" '
                        msg += f'to {mol_type} in order to use {method.__qualname__}'
                        print(msg)
                        self.mol_type = mol_type
                    return method(self, *args, **kwargs)
                return wrapper
            return decorator

    def __init__(self,
                 mol: Union[PybelMol, RDKitMol] = None,
                 smiles: str = None,
                 path: Union[str, Path] = None,
                 mol_type: str = 'rdkit',
                 name: str = None) -> None:
        """
        Args:
            mol: Molecule to be wrapped.
            smiles: SMILES used to create molecule.
            path: Path containing molecule.
            mol_type: Underlying molecule type to use (PybelMol or RDKitMol).
            name: Name of molecule.
        """
        num_not_none = sum(arg is not None for arg in [mol, smiles, path])
        if num_not_none != 1:
            raise ValueError('Must specify only one of mol, smiles, or path')

        self.__mol = None  # Required for mol_type setter
        self.mol_type = mol_type

        if mol is not None:
            self._mol = mol

        if smiles is not None:
            self._mol = PybelMol(smiles=smiles) if self.mol_type == 'pybel' else RDKitMol(smiles=smiles)

        if path is not None:
            self._mol = file_to_pybel(path) if self.mol_type == 'pybel' else file_to_rdkit(path)

        self.name = name

    @property
    def mol_type(self) -> str:
        return self._mol_type

    @mol_type.setter
    def mol_type(self, val: str) -> None:
        """Update self._mol each time mol_type is changed"""
        self._mol_type = process_mol_type(val)
        if self.__mol is not None:
            self._mol = self._mol  # Triggers setter

    @property
    def _mol(self) -> Union[PybelMol, RDKitMol]:
        return self.__mol

    @_mol.setter
    def _mol(self, val: Union[PybelMol, RDKitMol]) -> None:
        # Check pybel.Molecule and Chem.Mol so that PybelMol and RDKitMol can be instantiated from them, if necessary
        if isinstance(val, pybel.Molecule):  # Also true for PybelMol
            self.__mol = PybelMol(val)
            if self.mol_type == 'rdkit':
                self.__mol = pybel_to_rdkit(self.__mol)
        elif isinstance(val, Chem.Mol):  # Also true for RDKitMol
            self.__mol = RDKitMol(val)
            if self.mol_type == 'pybel':
                self.__mol = rdkit_to_pybel(self.__mol)
        else:
            raise ValueError(f'Invalid molecule of type "{type(val)}"')

        self._add_attrs_and_methods()

    def _add_attrs_and_methods(self) -> None:  # TODO: Does this do what I want it to do?
        """Add attributes and methods from PybelMol or RDKitMol to instance"""
        dunder_methods = {'__len__', '__iter__'}

        for attr in dir(self.__mol):
            if not attr.startswith('__') or attr in dunder_methods:
                setattr(self, attr, getattr(self.__mol, attr))
