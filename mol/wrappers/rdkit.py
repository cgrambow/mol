from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PropertyMol import PropertyMol

from .base import AbstractMol

# Dictionary of RDKit functions for converting molecule to other formats
MOL_TO_STRING_CONVERTERS: dict[str, Callable[[Chem.Mol, ...], str]] = {
    'can': Chem.MolToSmiles,
    'cxsmiles': Chem.MolToCXSmiles,
    'fa': Chem.MolToFASTA,
    'fasta': Chem.MolToFASTA,
    'fsa': Chem.MolToFASTA,
    'helm': Chem.MolToHELM,
    'mdl': Chem.MolToMolBlock,
    'mol': Chem.MolToMolBlock,
    'pdb': Chem.MolToPDBBlock,
    'sd': Chem.SDWriter.GetText,
    'sdf': Chem.SDWriter.GetText,
    'smi': Chem.MolToSmiles,
    'smiles': Chem.MolToSmiles,
    'tpl': Chem.MolToTPLBlock,
    'v3000': Chem.MolToV3KMolBlock,
    'v3000mol': Chem.MolToV3KMolBlock,
    'v3k': Chem.MolToV3KMolBlock,
    'v3kmol': Chem.MolToV3KMolBlock,
    'xyz': Chem.MolToXYZBlock,
}

MOL_TO_FILE_CONVERTERS: dict[str, Callable[[Chem.Mol, ...], None]] = {
    'mdl': Chem.MolToMolFile,
    'mol': Chem.MolToMolFile,
    'pdb': Chem.MolToPDBFile,
    'tpl': Chem.MolToTPLFile,
    'v3000': Chem.MolToV3KMolFile,
    'v3000mol': Chem.MolToV3KMolFile,
    'v3k': Chem.MolToV3KMolFile,
    'v3kmol': Chem.MolToV3KMolFile,
    'xyz': Chem.MolToXYZFile,
}

MOL_TO_FILE_WRITERS: dict[str, type] = {
    'sd': Chem.SDWriter,
    'sdf': Chem.SDWriter,
}


class _RDKitMolMeta(type(PropertyMol), type(AbstractMol)):
    """Resolve metaclass conflict"""
    pass


class RDKitMol(PropertyMol, AbstractMol, metaclass=_RDKitMolMeta):
    """
    Wrapper for rdkit.Chem.Mol with properties.
    """
    def __init__(self, *args, smiles: str = None, **kwargs) -> None:
        """
        Initialize from SMILES string or from Chem.Mol.

        Args:
            *args: Arguments for Chem.Mol constructor if smiles is None, otherwise arguments for Chem.MolFromSmiles.
            smiles: SMILES string.
            **kwargs: Keyword arguments for Chem.Mol constructor if smiles is None, otherwise keyword arguments for
                      Chem.MolFromSmiles.
        """
        if smiles is None:
            super().__init__(*args, **kwargs)
        else:
            mol = Chem.MolFromSmiles(smiles, *args, **kwargs)
            super().__init__(mol)

    def __len__(self) -> int:
        return self.GetNumAtoms()

    def __iter__(self) -> Iterator[Chem.Atom]:
        for atom in self.GetAtoms():
            yield atom

    @property
    def name(self) -> str:
        try:
            return self.get_prop('_Name')
        except KeyError:
            return ''

    @name.setter
    def name(self, val: str) -> None:
        self.set_prop('_Name', val)

    @property
    def smiles(self) -> str:
        """Canonical SMILES"""
        return self.to('smi')

    @property
    def symbols(self) -> list[str]:
        """List of atomic symbols"""
        return [atom.GetSymbol() for atom in self]

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates"""
        return self.GetConformer().GetPositions()

    @coords.setter
    def coords(self, val: Union[np.ndarray, tuple[Sequence[str], np.ndarray]]) -> None:
        """
        Update coordinates and optionally check that atomic symbols match.

        Args:
            val: Numpy array of just the coordinates or tuple of symbols and coordinates.
        """
        symbols, coords = (None, val) if isinstance(val, np.ndarray) else val
        conformer = self.GetConformer()

        for i, (atom, xyz) in enumerate(zip(self, coords)):
            atom_idx = atom.GetIdx()

            if symbols is not None:
                symbol = atom.GetSymbol()
                if symbol != symbols[i]:
                    raise ValueError(f'Symbol {symbols[i]} does not match atom {atom_idx} of type {symbol}')

            conformer.SetAtomPosition(atom_idx, xyz)

    def copy(self) -> RDKitMol:
        return type(self)(self)

    def merge(self, other: RDKitMol) -> RDKitMol:
        """
        Combine self and other into new molecule.

        Args:
            other: Other RDKitMol.

        Returns:
            Merged RDKitMol.
        """
        new_mol = Chem.CombineMols(self, other)
        return type(self)(new_mol)

    def addh(self) -> None:
        """
        Add hydrogens
        """
        new_mol = Chem.AddHs(self)
        self.__init__(new_mol)  # Updates self

    def get_prop(self, name: str) -> Any:
        """
        Get property stored in Chem.Mol.

        Args:
            name: Name of property.

        Returns:
            Value of property.
        """
        return self.GetProp(name)

    def set_prop(self, name: str, val: Any) -> None:
        """
        Set property of Chem.Mol.

        Args:
            name: Name of property.
            val: Value of property.
        """
        self.SetProp(name, val)

    def to(self, frmt: str, **kwargs) -> str:
        """
        Convert to string representation.

        Args:
            frmt: Format.
            **kwargs: Other keyword arguments for conversion function.

        Returns:
            String representation in new format.
        """
        try:
            converter = MOL_TO_STRING_CONVERTERS[frmt.lower()]
        except KeyError:
            raise ValueError(f'{frmt} is not a recognized RDKit format')
        else:
            return converter(self, **kwargs)

    def save(self, path: Union[str, Path], frmt: str = None, **kwargs) -> None:
        """
        Write to file. Infer format from path extension if frmt is None.

        Args:
            path: File path.
            frmt: File format.
            **kwargs: Other keyword arguments for conversion function.
        """
        if frmt is None:
            path = Path(path)
            frmt = path.suffix[1:]
        path = str(path)

        try:
            converter = MOL_TO_FILE_CONVERTERS[frmt.lower()]
        except KeyError:
            try:
                writer_class = MOL_TO_FILE_WRITERS[frmt.lower()]
            except KeyError:
                raise ValueError(f'{frmt} is not a recognized RDKit format')
            else:
                writer = writer_class(path)  # Only takes file name as its argument
                writer.write(self)
                writer.close()
        else:
            converter(self, path, **kwargs)

    def generate_3d_geometry(self, forcefield: str = 'mmff94', nconf: int = 1) -> None:
        """
        Generate conformers using ETKDG method. Adds explicit hydrogens
        if they don't already exist.

        Args:
            forcefield: Force field type (only supports MMFF94).
            nconf: Number of conformers to embed.
        """
        if forcefield.lower() != 'mmff94':
            raise NotImplementedError('RDKit can only use MMFF94 force field')

        self.addh()

        if nconf == 1:
            AllChem.EmbedMolecule(self)
        else:
            AllChem.EmbedMultipleConfs(self, numConfs=nconf)

    def optimize_geometry(self, forcefield: str = 'mmff94', steps: int = 500) -> None:
        """
        Optimizer conformer geometries using force field energy
        minimization and retain the lowest energy conformer.

        Args:
            forcefield: Force field type (only supports MMFF94).
            steps: Maximum number of iterations.
        """
        if forcefield.lower() != 'mmff94':
            raise NotImplementedError('RDKit can only use MMFF94 force field')

        if self.GetNumConformers() == 1:
            AllChem.MMFFOptimizeMolecule(self, maxIters=steps)
        else:
            res = AllChem.MMFFOptimizeMoleculeConfs(self, maxIters=steps)
            energies = np.array(res)[:, 1]
            min_idx = np.argmin(energies)
            conf = list(self.GetConformers())[min_idx]
            new_mol = Chem.Mol(self)
            new_mol.RemoveAllConformers()
            new_mol.AddConformer(conf)
            self.__init__(new_mol)  # Updates self

    def enumerate_double_bond_stereoisomers(self, reset_stereo: bool = True) -> Iterator[RDKitMol]:
        """
        Enumerate all double-bond stereoisomers.

        Args:
            reset_stereo: Remove existing double-bond stereochemistry first so that all bonds are considered
                          (destroys 3D geometry and atom order)

        Returns:
            Generator of unique steroisomers
        """
        mol = self.copy()
        if reset_stereo:
            mol.remove_stereo()

        for bond in mol.GetBonds():
            if bond.GetBondDir() == Chem.BondDir.EITHERDOUBLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        Chem.FindPotentialStereoBonds(mol)
        stereo_bonds = [bond for bond in mol.GetBonds() if bond.GetStereo() == Chem.BondStereo.STEREOANY]
        if not stereo_bonds:
            yield self  # Don't return the copy
            return

        smiles = set()
        nstereo = len(stereo_bonds)

        for bitflag in range(2 ** nstereo):  # Each bond has two states
            # Set unique stereo combination
            for i in range(nstereo):
                if bool(bitflag & (1 << i)):
                    stereo_bonds[i].SetStereo(Chem.BondStereo.STEREOCIS)
                else:
                    stereo_bonds[i].SetStereo(Chem.BondStereo.STEREOTRANS)

            isomer = mol.copy()
            Chem.SetDoubleBondNeighborDirections(isomer)
            isomer.ClearComputedProps(includeRings=False)
            Chem.AssignStereochemistry(isomer, cleanIt=True, force=True, flagPossibleStereoCenters=True)

            # Make sure there are no accidental duplicates
            smi = Chem.MolToSmiles(isomer, isomericSmiles=True)
            if smi in smiles:
                continue
            smiles.add(smi)

            yield isomer

    def remove_stereo(self) -> None:
        """
        Remove stereochemistry. Destroys 3D geometry and atom order.
        """
        smi = self.to('smi', isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
        self.__init__(mol)
