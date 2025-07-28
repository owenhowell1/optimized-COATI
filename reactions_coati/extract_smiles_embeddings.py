#!/usr/bin/env python3
"""
Extract all SMILES strings from reactions and compute their COATI embeddings
"""

import os
import pickle
import torch
import pandas as pd
from rdkit import Chem
from typing import List, Dict

# Import COATI components
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from coati.generative.coati_purifications import embed_smiles

def load_reactions(dataset_name: str) -> List[Dict]:
    """Load reactions from dataset"""
    filepath = f'datasets/{dataset_name}.pkl'
    if not os.path.exists(filepath):
        print(f"Dataset {filepath} not found!")
        return []
    
    with open(filepath, 'rb') as f:
        reactions = pickle.load(f)
    
    print(f"Loaded {len(reactions)} reactions from {dataset_name}")
    return reactions

def extract_all_smiles(reactions: List[Dict]) -> List[str]:
    """Extract all unique SMILES strings from reactions"""
    smiles_set = set()
    
    for reaction in reactions:
        reactant = reaction['reactant_smiles']
        product = reaction['product_smiles']
        
        # Validate SMILES
        if Chem.MolFromSmiles(reactant) is not None:
            smiles_set.add(reactant)
        if Chem.MolFromSmiles(product) is not None:
            smiles_set.add(product)
    
    smiles_list = list(smiles_set)
    print(f"Extracted {len(smiles_list)} unique SMILES strings")
    return smiles_list

def compute_embeddings(smiles_list: List[str]) -> pd.DataFrame:
    """Compute COATI embeddings for all SMILES strings"""
    print("Loading COATI model...")
    encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
        freeze=True,
        device="cpu",
        doc_url="s3://terray-public/models/grande_closed.pkl"
    )
    print("COATI model loaded successfully!")
    
    results = []
    
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processing SMILES {i}/{len(smiles_list)}")
        
        try:
            # Compute embedding
            embedding = embed_smiles(smiles, encoder, tokenizer)
            
            # Store result
            result = {
                'smiles': smiles,
                'embedding_norm': torch.norm(embedding).item(),
                'embedding_dim': embedding.shape[0]
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error embedding {smiles}: {e}")
            continue
    
    return pd.DataFrame(results)

def main():
    """Main function"""
    print("=== SMILES Embedding Extraction ===\n")
    
    # Load reactions
    reactions = load_reactions('synthetic_reactions')
    if not reactions:
        return
    
    # Extract all SMILES
    smiles_list = extract_all_smiles(reactions)
    
    # Compute embeddings
    embeddings_df = compute_embeddings(smiles_list)
    
    # Save results
    output_file = 'smiles_embeddings.csv'
    embeddings_df.to_csv(output_file, index=False)
    print(f"\nEmbeddings saved to {output_file}")
    print(f"Processed {len(embeddings_df)} SMILES strings")

if __name__ == "__main__":
    main() 