import io
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Union

from openbabel import pybel
from rdkit import Chem

from .wrappers import PybelMol, RDKitMol
from .wrappers.rdkit import MOL_TO_FILE_WRITERS as RDKIT_MOL_TO_FILE_WRITERS


def _smiles_to_rdkit(smiles: str, **kwargs) -> Chem.Mol:
    """Chem.MolFromSmiles wrapper so that signature matches other converters"""
    params = Chem.SmilesParserParams()
    for k, v in kwargs.items():
        setattr(params, k, v)
    return Chem.MolFromSmiles(smiles, params)


def _sdf_string_to_rdkit(sdf: str, **kwargs) -> Chem.Mol:
    """Helper function to convert SDF string to RDKit molecule with properties"""
    stream = io.BytesIO(sdf.encode('utf-8'))  # ForwardSDMolSupplier requires binary object
    supplier = Chem.ForwardSDMolSupplier(stream, **kwargs)
    return next(supplier)  # Assumes that SDF string just contains one molecule


RDKIT_STRING_TO_MOL_CONVERTERS: dict[str, Callable[[str, ...], Chem.Mol]] = {
    'can': _smiles_to_rdkit,
    'mdl': Chem.MolFromMolBlock,
    'mol': Chem.MolFromMolBlock,
    'mol2': Chem.MolFromMol2Block,
    'pdb': Chem.MolFromPDBBlock,
    'sd': _sdf_string_to_rdkit,
    'sdf': _sdf_string_to_rdkit,
    'smi': _smiles_to_rdkit,
    'smiles': _smiles_to_rdkit,
    'v3000': Chem.MolFromMolBlock,
    'v3000mol': Chem.MolFromMolBlock,
    'v3k': Chem.MolFromMolBlock,
    'v3kmol': Chem.MolFromMolBlock,
}

RDKIT_FILE_TO_MOL_CONVERTERS: dict[str, Callable[[str, ...], Chem.Mol]] = {
    'mdl': Chem.MolFromMolFile,
    'mol': Chem.MolFromMolFile,
    'mol2': Chem.MolFromMol2File,
    'pdb': Chem.MolFromPDBFile,
}

RDKIT_FILE_TO_MOL_SUPPLIERS: dict[str, type] = {
    'sd': Chem.SDMolSupplier,
    'sdf': Chem.SDMolSupplier,
}


def string_to_pybel(frmt: str, string: str, **kwargs) -> PybelMol:
    """
    Convert string representation of molecule to PybelMol.

    Args:
        frmt: Format of string.
        string: String representation of molecule.
        **kwargs: Options for pybel.readstring.

    Returns:
        PybelMol corresponding to string representation.
    """
    pybel_mol = pybel.readstring(frmt, string, opt=kwargs)
    return PybelMol(pybel_mol)


def string_to_rdkit(frmt: str, string: str, **kwargs) -> RDKitMol:
    """
    Convert string representation of molecule to RDKitMol.

    Args:
        frmt: Format of string.
        string: String representation of molecule.
        **kwargs: Other keyword arguments for conversion function.

    Returns:
        RDKitMol corresponding to string representation.
    """
    try:
        converter = RDKIT_STRING_TO_MOL_CONVERTERS[frmt.lower()]
    except KeyError:
        raise ValueError(f'{frmt} is not a recognized RDKit format')
    else:
        remove_hs = kwargs.pop('removeHs', False)  # Don't remove hydrogens by default
        rdkit_mol = converter(string, removeHs=remove_hs, **kwargs)
        return RDKitMol(rdkit_mol)


def file_to_pybel(path: Union[str, Path], frmt: str = None, **kwargs) -> Union[PybelMol, list[PybelMol]]:
    """
    Convert file to PybelMol or list of PybelMols.
    Infer format from path extension if frmt is None.

    Args:
        path: Path to file.
        frmt: File format.
        **kwargs: Other keyword arguments for pybel.readfile.

    Returns:
        PybelMol if there is a single molecule in the file, otherwise list of PybelMols.
    """
    if frmt is None:
        path = Path(path)
        frmt = path.suffix[1:]

    mols = [PybelMol(m) for m in pybel.readfile(frmt.lower(), str(path), opt=kwargs)]
    return mols[0] if len(mols) == 1 else mols


def file_to_rdkit(path: Union[str, Path], frmt: str = None, **kwargs) -> Union[RDKitMol, list[RDKitMol]]:
    """
        Convert file to RDKitMol or list of RDKitMols.
        Infer format from path extension if frmt is None.

        Args:
            path: Path to file.
            frmt: File format.
            **kwargs: Other keyword arguments for conversion function.

        Returns:
            RDKitMol if there is a single molecule in the file, otherwise list of RDKitMols.
        """
    if frmt is None:
        path = Path(path)
        frmt = path.suffix[1:]

    remove_hs = kwargs.pop('removeHs', False)  # Don't remove hydrogens by default

    try:
        converter = RDKIT_FILE_TO_MOL_CONVERTERS[frmt.lower()]
    except KeyError:
        try:
            supplier_class = RDKIT_FILE_TO_MOL_SUPPLIERS[frmt.lower()]
        except KeyError:
            raise ValueError(f'{frmt} is not a valid RDKit file format')
        else:
            supplier = supplier_class(str(path), removeHs=remove_hs, **kwargs)
            mols = [RDKitMol(m) for m in supplier]
            return mols[0] if len(mols) == 1 else mols
    else:
        rdkit_mol = converter(str(path), removeHs=remove_hs, **kwargs)
        return RDKitMol(rdkit_mol)


def pybel_mols_to_file(pybel_mols: Iterable[PybelMol], path: Union[str, Path], frmt: str = None, **kwargs) -> None:
    """
    Write several PybelMols to a file.
    Infer format from path extension if frmt is None.

    Args:
        pybel_mols: PybelMol instances.
        path: File path.
        frmt: File format.
        **kwargs: Other keyword arguments for pybel.Outputfile.
    """
    if frmt is None:
        path = Path(path)
        frmt = path.suffix[1:]

    overwrite = kwargs.pop('overwrite', False)  # Don't overwrite by default

    with pybel.Outputfile(frmt.lower(), filename=str(path), overwrite=overwrite, opt=kwargs) as f:
        for pybel_mol in pybel_mols:
            f.write(pybel_mol)


def rdkit_mols_to_file(rdkit_mols: Iterable[RDKitMol], path: Union[str, Path], frmt: str = None, **kwargs) -> None:
    """
    Write several RDKitMols to a file.
    Infer format from path extension if frmt is None.

    Args:
        rdkit_mols: RDKitMol instances.
        path: File path.
        frmt: File format.
        **kwargs: Used for compatibility only.
    """
    if frmt is None:
        path = Path(path)
        frmt = path.suffix[1:]

    try:
        writer_class = RDKIT_MOL_TO_FILE_WRITERS[frmt.lower()]
    except KeyError:
        raise ValueError(f'{frmt} is not a recognized RDKit format')
    else:
        writer = writer_class(str(path))
        for rdkit_mol in rdkit_mols:
            writer.write(rdkit_mol)
        writer.close()


def pybel_to_rdkit(pybel_mol: PybelMol) -> RDKitMol:
    """
    Convert PybelMol to RDKitMol.

    Args:
        pybel_mol: PybelMol instance.

    Returns:
        RDKitMol instance corresponding to PybelMol.
    """
    mol_block = pybel_mol.to('sdf')
    return string_to_rdkit('sdf', mol_block)


def rdkit_to_pybel(rdkit_mol: RDKitMol) -> PybelMol:
    """
    Convert RDKitMol to PybelMol.

    Args:
        rdkit_mol: RDKitMol instance.

    Returns:
        PybelMol instance corresponding to RDKitMol.
    """
    mol_block = rdkit_mol.to('sdf')
    return string_to_pybel('sdf', mol_block)
