"""Compute the property feature vector phi(m, x) used by the external field.

Order must match configs.model.external_field.properties.
All getters take an RDKit Mol and return a float.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED

from thermofrag.utils.sa_scorer import sa_score


def _logp(mol):
    return float(Crippen.MolLogP(mol))


def _qed(mol):
    return float(QED.qed(mol))


def _sa(mol):
    return sa_score(mol)


def _tpsa(mol):
    return float(Descriptors.TPSA(mol))


def _mw(mol):
    return float(Descriptors.MolWt(mol))


def _hba(mol):
    return float(Lipinski.NumHAcceptors(mol))


def _hbd(mol):
    return float(Lipinski.NumHDonors(mol))


def _rotb(mol):
    return float(Lipinski.NumRotatableBonds(mol))


REGISTRY = {
    "logP": _logp,
    "qed": _qed,
    "sa": _sa,
    "tpsa": _tpsa,
    "mw": _mw,
    "hba": _hba,
    "hbd": _hbd,
    "rotb": _rotb,
}


def compute_phi(mol: Chem.Mol, properties: list[str]) -> np.ndarray:
    return np.asarray([REGISTRY[p](mol) for p in properties], dtype=np.float32)
