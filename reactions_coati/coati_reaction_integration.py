#!/usr/bin/env python3
"""
Compute COATI embeddings for reaction inputs and outputs
"""

import os
import pickle
import torch
import numpy as np
from rdkit import Chem
import pandas as pd
from typing import List, Dict, Tuple

# Import COATI components
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from coati.generative.coati_purifications import embed_smiles
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer

class COATIReactionEmbedder:
    """
    Compute COATI embeddings for reaction inputs and outputs
    """
    
    def __init__(self, model_url: str = "s3://terray-public/models/grande_closed.pkl"):
        """
        Initialize COATI model for reaction embedding
        """
        print("Loading COATI model...")
        self.encoder, self.tokenizer = load_e3gnn_smiles_clip_e2e(
            freeze=True,
            device="cpu",  # Change to "cuda:0" for GPU
            doc_url=model_url
        )
        print("COATI model loaded successfully!")
    
    def embed_molecule(self, smiles: str) -> torch.Tensor:
        """
        Embed a molecule using COATI
        """
        try:
            return embed_smiles(smiles, self.encoder, self.tokenizer)
        except Exception as e:
            print(f"Error embedding {smiles}: {e}")
            return None
    
    def embed_reaction_pair(self, reactant_smiles: str, product_smiles: str) -> Dict:
        """
        Embed both reactant and product of a reaction
        """
        reactant_embedding = self.embed_molecule(reactant_smiles)
        product_embedding = self.embed_molecule(product_smiles)
        
        if reactant_embedding is None or product_embedding is None:
            return None
        
        # Calculate similarity
        similarity = torch.cosine_similarity(
            reactant_embedding.unsqueeze(0), 
            product_embedding.unsqueeze(0)
        ).item()
        
        return {
            'reactant_embedding': reactant_embedding,
            'product_embedding': product_embedding,
            'similarity': similarity,
            'embedding_difference': (product_embedding - reactant_embedding).detach()
        }
    
    def compute_reaction_embeddings(self, dataset_name: str) -> pd.DataFrame:
        """
        Compute COATI embeddings for all reactions in a dataset
        """
        print(f"Computing embeddings for dataset: {dataset_name}")
        
        # Load dataset
        filepath = f'datasets/{dataset_name}.pkl'
        if not os.path.exists(filepath):
            print(f"Dataset {filepath} not found!")
            return None
        
        with open(filepath, 'rb') as f:
            reactions = pickle.load(f)
        
        print(f"Loaded {len(reactions)} reactions")
        
        # Compute embeddings for each reaction
        embedding_results = []
        
        for i, reaction in enumerate(reactions):
            if i % 100 == 0:
                print(f"Processing reaction {i}/{len(reactions)}")
            
            reactant_smiles = reaction['reactant_smiles']
            product_smiles = reaction['product_smiles']
            
            # Validate SMILES
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if reactant_mol is None or product_mol is None:
                print(f"Skipping invalid reaction {i}: {reactant_smiles} -> {product_smiles}")
                continue
            
            # Get embeddings
            embedding_analysis = self.embed_reaction_pair(reactant_smiles, product_smiles)
            
            if embedding_analysis is None:
                print(f"Failed to embed reaction {i}: {reactant_smiles} -> {product_smiles}")
                continue
            
            # Store results
            result = {
                'reaction_id': reaction.get('reaction_id', f'reaction_{i}'),
                'reactant_smiles': reactant_smiles,
                'product_smiles': product_smiles,
                'reaction_type': reaction.get('reaction_type', 'unknown'),
                'similarity': embedding_analysis['similarity'],
                'embedding_norm': torch.norm(embedding_analysis['embedding_difference']).item(),
                'reactant_length': len(reactant_smiles),
                'product_length': len(product_smiles),
                'reactant_embedding_norm': torch.norm(embedding_analysis['reactant_embedding']).item(),
                'product_embedding_norm': torch.norm(embedding_analysis['product_embedding']).item()
            }
            
            embedding_results.append(result)
        
        return pd.DataFrame(embedding_results)
    
    def save_embeddings(self, df: pd.DataFrame, dataset_name: str):
        """
        Save embedding results to file
        """
        output_file = f'{dataset_name}_embeddings.csv'
        df.to_csv(output_file, index=False)
        print(f"Embeddings saved to {output_file}")
        
        # Also save as pickle for later use
        pickle_file = f'{dataset_name}_embeddings.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(df.to_dict('records'), f)
        print(f"Embeddings saved to {pickle_file}")

def main():
    """
    Main function to compute COATI embeddings for reactions
    """
    print("=== COATI Reaction Embedding Computation ===\n")
    
    # Initialize embedder
    embedder = COATIReactionEmbedder()
    
    # Process synthetic reactions
    print("1. Computing embeddings for synthetic reactions...")
    synthetic_embeddings = embedder.compute_reaction_embeddings('synthetic_reactions')
    if synthetic_embeddings is not None:
        print(f"Synthetic reactions embedding complete: {len(synthetic_embeddings)} reactions")
        print(f"Average similarity: {synthetic_embeddings['similarity'].mean():.3f}")
        print(f"Average embedding norm: {synthetic_embeddings['embedding_norm'].mean():.3f}")
        
        # Save results
        embedder.save_embeddings(synthetic_embeddings, 'synthetic_reactions')
    
    # Process retrosynthesis reactions
    print("\n2. Computing embeddings for retrosynthesis reactions...")
    retro_embeddings = embedder.compute_reaction_embeddings('retrosynthesis_synthetic')
    if retro_embeddings is not None:
        print(f"Retrosynthesis reactions embedding complete: {len(retro_embeddings)} reactions")
        print(f"Average similarity: {retro_embeddings['similarity'].mean():.3f}")
        print(f"Average embedding norm: {retro_embeddings['embedding_norm'].mean():.3f}")
        
        # Save results
        embedder.save_embeddings(retro_embeddings, 'retrosynthesis_synthetic')
    
    print("\n=== Embedding Computation Complete ===")
    print("Available files:")
    print("- synthetic_reactions_embeddings.csv")
    print("- synthetic_reactions_embeddings.pkl")
    print("- retrosynthesis_synthetic_embeddings.csv")
    print("- retrosynthesis_synthetic_embeddings.pkl")

if __name__ == "__main__":
    main() 