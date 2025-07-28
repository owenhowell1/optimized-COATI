# COATI Reaction Analysis

The goal of this module is to explore how COATI can be used to analyze and predict molecular reactions.

## Quick Start

### 1. Download Reaction Datasets

First, get the reaction datasets:

```bash
cd reactions
python download_reaction_datasets.py
```

This will download:
- **USPTO-50k**: 50,000 reactions from US patents (train/test/valid splits)
- **Synthetic reactions**: Curated set of common reaction types
- **Retrosynthesis data**: For reverse reaction prediction

### 2. Explore the Datasets

Analyze and visualize the downloaded datasets:

```bash
python explore_reaction_datasets.py
```

This will:
- Show statistics for each dataset
- Generate visualizations of example reactions
- Create a summary comparison table

### 3. Integrate with COATI

Use COATI to analyze reactions:

```bash
python coati_reaction_integration.py
```

This demonstrates:
- Embedding reaction reactants and products
- Predicting reaction outcomes
- Finding similar reactions in the dataset

## Available Datasets

### USPTO-50k
- **Source**: US Patent Office
- **Size**: ~50,000 reactions
- **Format**: Reactant SMILES → Product SMILES
- **Use case**: Large-scale reaction prediction

### Synthetic Reactions
- **Source**: Curated common reaction types
- **Size**: ~80 reactions
- **Types**: Hydrolysis, esterification, amidation, alkylation, etc.
- **Use case**: Testing and validation

### Retrosynthesis Data
- **Source**: USPTO processed for reverse prediction
- **Size**: ~50,000 reactions
- **Use case**: Backward reaction prediction

## COATI Integration Features

### 1. Reaction Embedding
- Embed both reactants and products using COATI
- Calculate similarity between reactant and product embeddings
- Analyze embedding differences to understand reaction transformations

### 2. Reaction Prediction
- Given a reactant, predict possible products
- Use COATI's generative capabilities with noise injection
- Rank candidates by similarity to reactant

### 3. Similar Reaction Search
- Find reactions similar to a query molecule
- Search across both reactants and products
- Rank by embedding similarity

## Example Usage

```python
from coati_reaction_integration import COATIReactionAnalyzer

# Initialize analyzer
analyzer = COATIReactionAnalyzer()

# Analyze a reaction dataset
analysis = analyzer.analyze_reaction_dataset('synthetic_reactions')

# Predict reaction outcomes
predictions = analyzer.predict_reaction_outcome("c1ccccc1", num_candidates=10)

# Find similar reactions
similar = analyzer.find_similar_reactions("c1ccccc1O", 'synthetic_reactions', top_k=5)
```

## Output Files

After running the scripts, you'll have:

- `reactions/datasets/`: Downloaded reaction datasets
- `reactions/dataset_summary.csv`: Comparison of all datasets
- `reactions/example_reaction_*.png`: Visualizations of example reactions
- `reactions/synthetic_reactions_analysis.csv`: COATI analysis results

## Next Steps

1. **Train reaction-specific models**: Fine-tune COATI on reaction data
2. **Reaction classification**: Use embeddings to classify reaction types
3. **Retrosynthesis**: Predict reactants from products
4. **Reaction optimization**: Optimize reaction conditions using embeddings

## Requirements

- COATI model (automatically downloaded)
- RDKit for molecular processing
- PyTorch for deep learning operations
- Pandas for data analysis
- Matplotlib for visualizations