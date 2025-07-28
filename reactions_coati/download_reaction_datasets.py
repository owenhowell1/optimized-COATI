#!/usr/bin/env python3
"""
Generate synthetic and retrosynthesis reaction datasets for use with COATI
"""

import os
import pandas as pd
from rdkit import Chem
import pickle
from typing import List, Dict


def create_synthetic_reaction_dataset():
    """
    Create a synthetic reaction dataset for testing
    """
    print("Creating synthetic reaction dataset...")
    reactions = [
        ("CC(C)Br", "CC(C)O", "hydrolysis"),
        ("c1ccccc1Br", "c1ccccc1O", "hydrolysis"),
        ("CC(C)O", "CC(C)OC(=O)C", "esterification"),
        ("c1ccccc1O", "c1ccccc1OC(=O)C", "esterification"),
        ("CC(C)O", "CC(C)NC(=O)C", "amidation"),
        ("c1ccccc1O", "c1ccccc1NC(=O)C", "amidation"),
        ("c1ccccc1", "c1ccccc1CC", "alkylation"),
        ("c1ccccc1", "c1ccccc1C(C)C", "alkylation"),
        ("c1ccccc1", "c1ccccc1C(=O)C", "acylation"),
        ("c1ccccc1", "c1ccccc1C(=O)CC", "acylation"),
        ("c1ccccc1", "c1ccccc1Cl", "halogenation"),
        ("c1ccccc1", "c1ccccc1Br", "halogenation"),
        ("c1ccccc1", "c1ccccc1[N+](=O)[O-]", "nitration"),
        ("c1ccccc1", "c1ccccc1S(=O)(=O)O", "sulfonation"),
        ("c1ccccc1", "c1ccccc1CC", "addition"),
        ("c1ccccc1", "c1ccccc1C(C)C", "addition"),
        ("c1ccccc1Br", "c1ccccc1N", "substitution"),
        ("c1ccccc1Cl", "c1ccccc1O", "substitution"),
        ("c1ccccc1", "c1ccccc1O", "oxidation"),
        ("c1ccccc1", "c1ccccc1C(=O)O", "oxidation"),
        ("c1ccccc1C(=O)O", "c1ccccc1CO", "reduction"),
        ("c1ccccc1C(=O)C", "c1ccccc1CCO", "reduction"),
    ]
    synthetic_data = []
    for reactant, product, reaction_type in reactions:
        for i in range(10):
            if i % 3 == 0:
                var_reactant = reactant.replace("c1ccccc1", "c1ccc(C)cc1")
                var_product = product.replace("c1ccccc1", "c1ccc(C)cc1")
            elif i % 3 == 1:
                var_reactant = reactant.replace("c1ccccc1", "c1ccc(O)cc1")
                var_product = product.replace("c1ccccc1", "c1ccc(O)cc1")
            else:
                var_reactant = reactant.replace("c1ccccc1", "c1ccc(N)cc1")
                var_product = product.replace("c1ccccc1", "c1ccc(N)cc1")
            synthetic_data.append({
                'reactant_smiles': var_reactant,
                'product_smiles': var_product,
                'reaction_type': reaction_type,
                'reaction_id': f"{reaction_type}_{i}",
                'variation': i
            })
    return pd.DataFrame(synthetic_data)


def create_synthetic_retrosynthesis_data(num_reactions: int = 5000):
    """
    Create synthetic retrosynthesis data
    """
    print(f"Creating synthetic retrosynthesis data with {num_reactions} reactions...")
    retro_templates = [
        ("CC(C)OC(=O)C", "CC(C)O", "reverse_esterification"),
        ("c1ccccc1OC(=O)C", "c1ccccc1O", "reverse_esterification"),
        ("CC(C)NC(=O)C", "CC(C)O", "reverse_amidation"),
        ("c1ccccc1NC(=O)C", "c1ccccc1O", "reverse_amidation"),
        ("c1ccccc1CC", "c1ccccc1", "reverse_alkylation"),
        ("c1ccccc1C(C)C", "c1ccccc1", "reverse_alkylation"),
        ("c1ccccc1Cl", "c1ccccc1", "reverse_halogenation"),
        ("c1ccccc1Br", "c1ccccc1", "reverse_halogenation"),
        ("c1ccccc1O", "c1ccccc1", "reverse_oxidation"),
        ("c1ccccc1C(=O)O", "c1ccccc1", "reverse_oxidation"),
        ("c1ccccc1CO", "c1ccccc1C(=O)O", "reverse_reduction"),
        ("c1ccccc1CCO", "c1ccccc1C(=O)C", "reverse_reduction"),
    ]
    synthetic_data = []
    for i in range(num_reactions):
        product, reactant, reaction_type = retro_templates[i % len(retro_templates)]
        if i % 4 == 0:
            product = product.replace("c1ccccc1", "c1ccc(C)cc1")
        elif i % 4 == 1:
            product = product.replace("c1ccccc1", "c1ccc(O)cc1")
        elif i % 4 == 2:
            product = product.replace("c1ccccc1", "c1ccc(N)cc1")
        synthetic_data.append({
            'product_smiles': product,
            'reactant_smiles': reactant,
            'reaction_type': reaction_type,
            'reaction_id': f"retro_synthetic_{i}",
            'difficulty': 'easy' if i % 3 == 0 else 'medium' if i % 3 == 1 else 'hard'
        })
    return pd.DataFrame(synthetic_data)


def process_reaction_data(df: pd.DataFrame) -> List[Dict]:
    processed_data = []
    for idx, row in df.iterrows():
        try:
            reactant = row['reactant_smiles']
            product = row['product_smiles']
            if pd.isna(reactant) or pd.isna(product):
                continue
            reactant_mol = Chem.MolFromSmiles(reactant)
            product_mol = Chem.MolFromSmiles(product)
            if reactant_mol is None or product_mol is None:
                continue
            reaction_smiles = f"{reactant}>>{product}"
            processed_data.append({
                'reactant_smiles': reactant,
                'product_smiles': product,
                'reaction_smiles': reaction_smiles,
                'reaction_id': row.get('reaction_id', f'reaction_{idx}'),
                'reaction_type': row.get('reaction_type', 'unknown'),
                'variation': row.get('variation', 0),
                'difficulty': row.get('difficulty', 'medium')
            })
        except Exception as e:
            print(f"Error processing reaction {idx}: {e}")
            continue
    return processed_data


def save_dataset(data: List[Dict], filename: str):
    os.makedirs('datasets', exist_ok=True)
    with open(f'datasets/{filename}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {len(data)} reactions to datasets/{filename}.pkl")


def main():
    print("=== Synthetic Reaction Dataset Generator ===\n")
    os.makedirs('datasets', exist_ok=True)
    print("1. Creating synthetic reaction dataset...")
    synthetic_df = create_synthetic_reaction_dataset()
    synthetic_processed = process_reaction_data(synthetic_df)
    save_dataset(synthetic_processed, 'synthetic_reactions')
    print("\n2. Creating retrosynthesis datasets...")
    retro_data = create_synthetic_retrosynthesis_data(5000)
    retro_processed = process_reaction_data(retro_data)
    save_dataset(retro_processed, 'retrosynthesis_synthetic')
    print("\n=== Dataset Generation Complete ===")
    print("Available datasets:")
    print("- datasets/synthetic_reactions.pkl")
    print("- datasets/retrosynthesis_synthetic.pkl")

if __name__ == "__main__":
    main() 