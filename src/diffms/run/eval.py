import pickle
import os
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit.Chem import (
    DataStructs,
    Draw,
    MolFromSmiles,
    rdFingerprintGenerator,
    rdMolHash,
)
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolHash import HashFunction

from diffms import ROOT
from diffms.utils import is_valid, mol2smiles, tanimoto_sim


@dataclass
class MolCandidate:
    smile: str
    num_gen: int
    true_smile: str

    @property
    def mol(self):
        return MolFromSmiles(self.smile)

    @property
    def tan_sim(self):
        true_mol = MolFromSmiles(self.true_smile)
        return tanimoto_sim_mol(self.mol, true_mol)

    @property
    def num_atoms(self):
        return self.mol.GetNumHeavyAtoms()

    @property
    def formula(self):
        return rdMolHash.MolHash(self.mol, HashFunction.MolFormula)

    def GetDrawing(self, **kwargs):
        return Draw.MolToImage(self.mol, **kwargs)

    def __eq__(self, other):
        return (self.num_gen == other.num_gen) & (self.tan_sim == other.tan_sim)

    def __gt__(self, other):
        if self.num_gen != other.num_gen:
            return self.num_gen > other.num_gen

        else:
            return self.tan_sim > other.tan_sim

    def __str__(self):
        return self.smile

    def __repr__(self):
        return f"(Formula : {self.formula} | Smile : {self.smile})"


def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument("--folder", required=True, type=str)

    return parser


def mol_to_fingerprint(m: Mol, radius: int = 3, nbits: int = 2048):
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    curr_fp = morgan_gen.GetFingerprint(m)
    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)

    return fingerprint


def tanimoto_sim_mol(mol1: Mol, mol2: Mol):
    fp1 = mol_to_fingerprint(mol1)
    fp2 = mol_to_fingerprint(mol2)

    return tanimoto_sim(fp1, fp2)


def evaluate_pred(true_mol, preds, verbose=False):
    # Count the number of valid preds
    valid_preds = [p for p in preds if is_valid(p)]
    num_valid = len(valid_preds)

    # Evaluate the number of time each molecule appears
    candidate_mols = Counter(mol2smiles(p) for p in valid_preds)
    best_gen = candidate_mols.most_common(1)[0][1]

    # List the candidate mols
    list_mols = [
        MolCandidate(smile, num_gen, mol2smiles(true_mol))
        for (smile, num_gen) in candidate_mols.items()
    ]
    sorted_mols = sorted(list_mols, reverse=True)

    # Select the 10 bests
    candidates = sorted_mols[: min(10, len(sorted_mols))]
    tan1 = candidates[0].tan_sim
    tan10 = max(m.tan_sim for m in candidates)

    if verbose:
        smile_len = max(len(m.smile) for m in sorted_mols[:10]) + 2
        labels = ["SMILE", "formula", "Generations", "Tanimoto Sim"]
        print(
            f"{labels[0]:{smile_len}} | {labels[1]:<15} | {labels[2]:<12} | {labels[3]:<10}"
        )

        for mol in sorted_mols[:10]:
            print(
                f"{mol.smile:<{smile_len}} | {mol.formula:<15} | {mol.num_gen:<12} | {mol.tan_sim:<10.3f}"
            )

    metrics = {
        "num_valid": num_valid,
        "max_num_gen": best_gen,
        "tan@1": tan1,
        "tan@10": tan10,
    }

    return (metrics, candidates)


def main():
    parser = get_args_parser()
    args = parser.parse_args()

    # Open the folder with predicted and true mols
    if os.path.isdir(args.folder):
        path = Path(args.folder)
    else:
        path = ROOT / args.folder

    # Open the pickle files
    with open(path / "true_pkl", "rb") as f:
        true_mols = pickle.load(f)

    with open(path / "pred.pkl", "rb") as f:
        pred_mols = pickle.load(f)

    #
