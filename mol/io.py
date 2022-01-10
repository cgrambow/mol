from pathlib import Path
from typing import Union

from .conversion import file_to_pybel, file_to_rdkit, pybel_mols_to_file, rdkit_mols_to_file
from .mol import Mol, process_mol_type


def file_to_mol(path: Union[str, Path], mol_type: str = 'pybel', frmt: str = None, **kwargs) -> Union[Mol, list[Mol]]:
    """
    Convert file to mol or list of mols.
    Infer format from path extension if frmt is None.

    Args:
        path: Path to file.
        mol_type: Type of cheminformatics wrapper to use.
        frmt: File format.
        **kwargs: Other keyword arguments for conversion function.

    Returns:
        Mol if there is a single molecule in the file, otherwise list of Mols.
    """
    reader = file_to_pybel if process_mol_type(mol_type) == 'pybel' else file_to_rdkit
    mols = reader(path, frmt=frmt, **kwargs)
    return [Mol(mol_type=mol_type, mol=m) for m in mols] if isinstance(mols, list) else Mol(mol_type=mol_type, mol=mols)


def mols_to_file(mols: list[Mol], path: Union[str, Path], frmt: str = None, **kwargs) -> None:
    """
    Write several mols to a file.
    Infer format from path extension if frmt is None.

    Args:
        mols: List of mol instances.
        path: Path to file.
        frmt: File format.
        **kwargs: Other keyword arguments for writer class.
    """
    mol_type = mols[0].mol_type
    if any(m.mol_type != mol_type for m in mols):
        raise TypeError('All molecules must have the same mol_type')

    writer = pybel_mols_to_file if mol_type == 'pybel' else rdkit_mols_to_file
    writer((m._mol for m in mols), path, frmt=frmt, **kwargs)
