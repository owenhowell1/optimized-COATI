#!/usr/bin/env python3
"""
Explore and analyze molecular reaction datasets
"""

import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def load_dataset(filename: str):
    """
    Load a reaction dataset
    """
    filepath = f'datasets/{filename}.pkl'
    if not os.path.exists(filepath):
        print(f"Dataset {filepath} not found. Run download_reaction_datasets.py first.")
        return None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} reactions from {filename}")
    return data

def analyze_reaction_dataset(data: list, dataset_name: str):
    """
    Analyze a reaction dataset
    """
    print(f"\n=== Analysis of {dataset_name} ===")
    
    # Basic statistics
    print(f"Total reactions: {len(data)}")
    
    # Reaction types
    reaction_types = [r.get('reaction_type', 'unknown') for r in data]
    type_counts = Counter(reaction_types)
    print(f"Reaction types: {len(type_counts)}")
    print("Top 10 reaction types:")
    for rtype, count in type_counts.most_common(10):
        print(f"  {rtype}: {count}")
    
    # SMILES length analysis
    reactant_lengths = [len(r['reactant_smiles']) for r in data]
    product_lengths = [len(r['product_smiles']) for r in data]
    
    print(f"\nSMILES length statistics:")
    print(f"  Reactants - Mean: {np.mean(reactant_lengths):.1f}, Std: {np.std(reactant_lengths):.1f}")
    print(f"  Products - Mean: {np.mean(product_lengths):.1f}, Std: {np.std(product_lengths):.1f}")
    
    # Valid SMILES check
    valid_reactants = 0
    valid_products = 0
    valid_reactions = 0
    
    for reaction in data:
        reactant_mol = Chem.MolFromSmiles(reaction['reactant_smiles'])
        product_mol = Chem.MolFromSmiles(reaction['product_smiles'])
        
        if reactant_mol is not None:
            valid_reactants += 1
        if product_mol is not None:
            valid_products += 1
        if reactant_mol is not None and product_mol is not None:
            valid_reactions += 1
    
    print(f"\nSMILES validity:")
    print(f"  Valid reactants: {valid_reactants}/{len(data)} ({100*valid_reactants/len(data):.1f}%)")
    print(f"  Valid products: {valid_products}/{len(data)} ({100*valid_products/len(data):.1f}%)")
    print(f"  Valid reactions: {valid_reactions}/{len(data)} ({100*valid_reactions/len(data):.1f}%)")
    
    return {
        'total_reactions': len(data),
        'valid_reactions': valid_reactions,
        'reaction_types': len(type_counts),
        'avg_reactant_length': np.mean(reactant_lengths),
        'avg_product_length': np.mean(product_lengths)
    }

def visualize_reactions(data: list, dataset_name: str, num_examples: int = 5):
    """
    Visualize some example reactions
    """
    print(f"\n=== Example Reactions from {dataset_name} ===")
    
    # Get valid reactions
    valid_reactions = []
    for reaction in data:
        reactant_mol = Chem.MolFromSmiles(reaction['reactant_smiles'])
        product_mol = Chem.MolFromSmiles(reaction['product_smiles'])
        if reactant_mol is not None and product_mol is not None:
            valid_reactions.append(reaction)
    
    if len(valid_reactions) == 0:
        print("No valid reactions found!")
        return
    
    # Show examples
    for i, reaction in enumerate(valid_reactions[:num_examples]):
        print(f"\nExample {i+1}:")
        print(f"  Reactant: {reaction['reactant_smiles']}")
        print(f"  Product: {reaction['product_smiles']}")
        print(f"  Type: {reaction.get('reaction_type', 'unknown')}")
        
        # Create reaction SMILES
        reaction_smiles = f"{reaction['reactant_smiles']}>>{reaction['product_smiles']}"
        print(f"  Reaction SMILES: {reaction_smiles}")
        
        # Try to visualize
        try:
            reactant_mol = Chem.MolFromSmiles(reaction['reactant_smiles'])
            product_mol = Chem.MolFromSmiles(reaction['product_smiles'])
            
            # Create a combined image
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Draw reactant
            ax1.imshow(Draw.MolToImage(reactant_mol, size=(300, 300)))
            ax1.set_title('Reactant')
            ax1.axis('off')
            
            # Draw product
            ax2.imshow(Draw.MolToImage(product_mol, size=(300, 300)))
            ax2.set_title('Product')
            ax2.axis('off')
            
            plt.suptitle(f'Reaction {i+1}: {reaction.get("reaction_type", "unknown")}')
            plt.tight_layout()
            plt.savefig(f'example_reaction_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Visualization saved to example_reaction_{i+1}.png")
            
        except Exception as e:
            print(f"  Could not visualize: {e}")

def create_reaction_summary():
    """
    Create a summary of all available datasets
    """
    print("=== Reaction Dataset Summary ===\n")
    
    datasets = [
        'synthetic_reactions',
        'retrosynthesis_synthetic'
    ]
    
    summary_data = []
    
    for dataset_name in datasets:
        data = load_dataset(dataset_name)
        if data is not None:
            stats = analyze_reaction_dataset(data, dataset_name)
            stats['dataset_name'] = dataset_name
            summary_data.append(stats)
            
            # Show examples
            visualize_reactions(data, dataset_name, num_examples=3)
    
    # Create summary table
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\n=== Dataset Comparison ===")
        print(df.to_string(index=False))
        
        # Save summary
        df.to_csv('dataset_summary.csv', index=False)
        print("\nSummary saved to dataset_summary.csv")

def main():
    """
    Main function to explore reaction datasets
    """
    print("=== Reaction Dataset Explorer ===\n")
    
    # Check if datasets exist
    if not os.path.exists('datasets'):
        print("No datasets found. Please run download_reaction_datasets.py first.")
        return
    
    # Create summary
    create_reaction_summary()
    
    print("\n=== Exploration Complete ===")
    print("Available files:")
    print("- dataset_summary.csv")
    print("- example_reaction_*.png")

if __name__ == "__main__":
    main() 