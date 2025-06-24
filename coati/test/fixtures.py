"""
Test fixtures for COATI data structures.

This module provides sample data for testing COATI functionality including:
- Molecular data with SMILES, atoms, coordinates
- Tokenized data for transformer models
- Batch data for training
- Various molecular representations
"""

import numpy as np
import torch
from typing import Dict, List, Any


# Basic molecular data fixtures
def get_simple_molecule_data() -> Dict[str, Any]:
    """Returns a simple benzene molecule data structure."""
    return {
        "smiles": "c1ccccc1",
        "atoms": np.array([6, 6, 6, 6, 6, 6], dtype=np.uint8),  # Carbon atoms
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.1, 1.2, 0.0],
                [1.4, 2.4, 0.0],
                [0.0, 2.4, 0.0],
                [-0.7, 1.2, 0.0],
            ],
            dtype=np.float32,
        ),
        "source_collection": "test_mols",
        "mod_molecule": 42,  # For partitioning
    }


def get_ethanol_data() -> Dict[str, Any]:
    """Returns ethanol molecule data structure."""
    return {
        "smiles": "CCO",
        "atoms": np.array([6, 6, 8], dtype=np.uint8),  # C, C, O
        "coords": np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.2, 1.0, 0.0]], dtype=np.float32
        ),
        "source_collection": "test_mols",
        "mod_molecule": 15,
    }


def get_aspirin_data() -> Dict[str, Any]:
    """Returns aspirin molecule data structure."""
    return {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "atoms": np.array(
            [6, 6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 8, 6, 8, 8], dtype=np.uint8
        ),
        "coords": np.array(
            [
                [0.0, 0.0, 0.0],  # C
                [1.5, 0.0, 0.0],  # C
                [2.2, 1.0, 0.0],  # O
                [1.5, 2.0, 0.0],  # C
                [2.2, 2.7, 0.0],  # O
                [0.8, 2.0, 0.0],  # C
                [0.0, 2.7, 0.0],  # C
                [-0.8, 2.0, 0.0],  # C
                [-1.5, 2.7, 0.0],  # C
                [-2.3, 2.0, 0.0],  # C
                [-2.3, 1.0, 0.0],  # C
                [-1.5, 0.3, 0.0],  # C
                [-1.5, -0.7, 0.0],  # O
                [-0.8, -1.4, 0.0],  # C
                [-0.8, -2.4, 0.0],  # O
                [0.2, -1.4, 0.0],  # O
            ],
            dtype=np.float32,
        ),
        "source_collection": "drugbank",
        "mod_molecule": 73,
    }


def get_batch_data(batch_size: int = 3) -> Dict[str, Any]:
    """Returns a batch of molecular data for testing."""
    molecules = [get_simple_molecule_data(), get_ethanol_data(), get_aspirin_data()]

    # Pad to batch_size if needed
    while len(molecules) < batch_size:
        molecules.append(get_simple_molecule_data())

    # Stack the data
    max_atoms = max(len(mol["atoms"]) for mol in molecules)

    atoms = np.zeros((batch_size, max_atoms), dtype=np.uint8)
    coords = np.zeros((batch_size, max_atoms, 3), dtype=np.float32)
    smiles = []
    source_collection = []
    mod_molecule = []

    for i, mol in enumerate(molecules):
        n_atoms = len(mol["atoms"])
        atoms[i, :n_atoms] = mol["atoms"]
        coords[i, :n_atoms, :] = mol["coords"]
        smiles.append(mol["smiles"])
        source_collection.append(mol["source_collection"])
        mod_molecule.append(mol["mod_molecule"])

    return {
        "atoms": atoms,
        "coords": coords,
        "smiles": smiles,
        "source_collection": source_collection,
        "mod_molecule": mod_molecule,
    }


def get_tokenized_batch_data(batch_size: int = 3) -> Dict[str, Any]:
    """Returns tokenized batch data for transformer testing."""
    batch_data = get_batch_data(batch_size)

    # Mock tokenized data (these would normally come from a tokenizer)
    tokens = []
    raw_tokens = []
    y_next = []

    for i in range(batch_size):
        # Mock token sequences
        seq_len = 20
        token_seq = np.random.randint(0, 1000, seq_len, dtype=np.int64)
        raw_token_seq = token_seq.copy()
        y_next_seq = np.roll(token_seq, -1)  # Shift by 1 for next token prediction
        y_next_seq[-1] = 0  # EOS token

        tokens.append(token_seq)
        raw_tokens.append(raw_token_seq)
        y_next.append(y_next_seq)

    # Stack into tensors
    batch_data.update(
        {
            "tokens": torch.tensor(tokens),
            "raw_tokens": torch.tensor(raw_tokens),
            "y_next": torch.tensor(y_next),
        }
    )

    return batch_data


def get_adjacency_matrix_data() -> Dict[str, Any]:
    """Returns molecular data with adjacency matrices."""
    # Benzene with adjacency matrix
    benzene_data = get_simple_molecule_data()

    # 6x6 adjacency matrix for benzene (ring structure)
    adj_mat = np.array(
        [
            [0, 1, 0, 0, 0, 1],  # C1 connected to C2 and C6
            [1, 0, 1, 0, 0, 0],  # C2 connected to C1 and C3
            [0, 1, 0, 1, 0, 0],  # C3 connected to C2 and C4
            [0, 0, 1, 0, 1, 0],  # C4 connected to C3 and C5
            [0, 0, 0, 1, 0, 1],  # C5 connected to C4 and C6
            [1, 0, 0, 0, 1, 0],  # C6 connected to C5 and C1
        ],
        dtype=np.int8,
    )

    benzene_data.update({"adj_mat": adj_mat, "adj_mat_atoms": benzene_data["atoms"]})

    return benzene_data


def get_morgan_fingerprint_data() -> Dict[str, Any]:
    """Returns molecular data with Morgan fingerprints."""
    mol_data = get_simple_molecule_data()

    # Mock 2048-bit Morgan fingerprint
    morgan_fp = np.random.randint(0, 2, 2048, dtype=np.uint8)

    mol_data.update({"morgan": morgan_fp})

    return mol_data


def get_selfies_data() -> Dict[str, Any]:
    """Returns molecular data with SELFIES representation."""
    mol_data = get_simple_molecule_data()

    # Mock SELFIES string (simplified)
    mol_data.update(
        {
            "selfies": "[C][C][C][C][C][C]",
            "rand_selfies": "[C][C][C][C][C][C]",  # Randomized version
        }
    )

    return mol_data


def get_property_data() -> Dict[str, Any]:
    """Returns molecular data with calculated properties."""
    mol_data = get_simple_molecule_data()

    mol_data.update(
        {
            "logp": 2.13,
            "qed": 0.85,
            "pic50": 6.2,
            "molecular_weight": 78.11,
            "num_atoms": 12,
            "num_rings": 1,
        }
    )

    return mol_data


def get_3d_conformer_data() -> Dict[str, Any]:
    """Returns molecular data with 3D conformer information."""
    mol_data = get_aspirin_data()

    # Add conformer-specific data
    mol_data.update(
        {
            "conformer_energy": -45.67,
            "rmsd": 0.23,
            "torsion_angles": np.array([60.0, 120.0, 180.0, 240.0, 300.0]),
            "bond_lengths": np.array(
                [
                    1.4,
                    1.5,
                    1.2,
                    1.4,
                    1.3,
                    1.4,
                    1.4,
                    1.4,
                    1.4,
                    1.4,
                    1.4,
                    1.4,
                    1.2,
                    1.5,
                    1.2,
                    1.2,
                ]
            ),
            "bond_angles": np.array(
                [
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                    120.0,
                ]
            ),
        }
    )

    return mol_data


def get_training_batch_data(batch_size: int = 4) -> Dict[str, Any]:
    """Returns a complete training batch with all necessary fields."""
    batch_data = get_batch_data(batch_size)

    # Add training-specific fields
    batch_data.update(
        {
            "tokens": torch.randint(0, 1000, (batch_size, 50)),
            "raw_tokens": torch.randint(0, 1000, (batch_size, 50)),
            "y_next": torch.randint(0, 1000, (batch_size, 50)),
            "clip_embeddings": torch.randn(batch_size, 256),
            "e3gnn_embeddings": torch.randn(batch_size, 256),
            "transformer_embeddings": torch.randn(batch_size, 256),
            "loss_mask": torch.ones(batch_size, 50, dtype=torch.bool),
            "attention_mask": torch.ones(batch_size, 50, dtype=torch.bool),
        }
    )

    return batch_data


def get_validation_data() -> List[Dict[str, Any]]:
    """Returns a list of validation molecules."""
    return [
        get_simple_molecule_data(),
        get_ethanol_data(),
        get_aspirin_data(),
        get_property_data(),
        get_3d_conformer_data(),
    ]


def get_test_data() -> List[Dict[str, Any]]:
    """Returns a list of test molecules."""
    return [get_simple_molecule_data(), get_ethanol_data(), get_aspirin_data()]


def get_dataset_config() -> Dict[str, Any]:
    """Returns configuration for COATI dataset."""
    return {
        "cache_dir": "./test_cache",
        "fields": ["smiles", "atoms", "coords"],
        "test_split_mode": "row",
        "test_frac": 0.02,
        "valid_frac": 0.02,
    }


def get_model_config() -> Dict[str, Any]:
    """Returns configuration for COATI model."""
    return {
        "n_layer_e3gnn": 4,
        "n_hidden_e3nn": 128,
        "msg_cutoff_e3nn": 10.0,
        "n_hidden_xformer": 128,
        "n_embd_common": 128,
        "n_layer_xformer": 16,
        "n_head": 8,
        "n_seq": 200,
        "biases": True,
        "torch_emb": False,
        "norm_clips": False,
        "norm_embed": False,
        "token_mlp": False,
    }


def get_tokenizer_config() -> Dict[str, Any]:
    """Returns configuration for tokenizer."""
    return {
        "n_seq": 200,
        "vocab_size": 1000,
        "special_tokens": [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[SMILES]",
            "[STOP]",
        ],
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "smiles_token": "[SMILES]",
        "stop_token": "[STOP]",
    }


def get_training_config() -> Dict[str, Any]:
    """Returns training configuration."""
    return {
        "batch_size": 32,
        "learning_rate": 4e-4,
        "weight_decay": 0.1,
        "clip_grad": 10.0,
        "n_epochs": 10,
        "warmup_steps": 1000,
        "scheduler": "cosine",
        "optimizer": "adam",
        "mixed_precision": True,
        "gradient_accumulation_steps": 1,
    }


# Utility functions for creating specific test scenarios
def create_molecule_with_properties(
    smiles: str, properties: Dict[str, float]
) -> Dict[str, Any]:
    """Creates a molecule data structure with specified properties."""
    from coati.containers.rdkit_utils import mol_to_atoms_coords

    try:
        atoms, coords = mol_to_atoms_coords(smiles)
        mol_data = {
            "smiles": smiles,
            "atoms": atoms,
            "coords": coords,
            "source_collection": "test_mols",
            "mod_molecule": hash(smiles) % 100,
        }
        mol_data.update(properties)
        return mol_data
    except:
        # Fallback to mock data if RDKit fails
        return get_simple_molecule_data()


def create_batch_with_specific_molecules(smiles_list: List[str]) -> Dict[str, Any]:
    """Creates a batch with specific SMILES strings."""
    molecules = []
    for smiles in smiles_list:
        mol_data = create_molecule_with_properties(smiles, {})
        molecules.append(mol_data)

    return get_batch_data_from_molecules(molecules)


def get_batch_data_from_molecules(molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Creates batch data from a list of molecule dictionaries."""
    batch_size = len(molecules)
    max_atoms = max(len(mol["atoms"]) for mol in molecules)

    atoms = np.zeros((batch_size, max_atoms), dtype=np.uint8)
    coords = np.zeros((batch_size, max_atoms, 3), dtype=np.float32)
    smiles = []
    source_collection = []
    mod_molecule = []

    for i, mol in enumerate(molecules):
        n_atoms = len(mol["atoms"])
        atoms[i, :n_atoms] = mol["atoms"]
        coords[i, :n_atoms, :] = mol["coords"]
        smiles.append(mol["smiles"])
        source_collection.append(mol.get("source_collection", "test_mols"))
        mod_molecule.append(mol.get("mod_molecule", i))

    return {
        "atoms": atoms,
        "coords": coords,
        "smiles": smiles,
        "source_collection": source_collection,
        "mod_molecule": mod_molecule,
    }


# Common test molecules
COMMON_SMILES = [
    "c1ccccc1",  # Benzene
    "CCO",  # Ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC1=C(C2=C(O1)C=C(C=C2)OC3=NC=NC4=CC(=C(C=C43)OC)OC)C(=O)NC",  # Fruquintinib
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # Phenylalanine
    "C1=CC=C(C=C1)CC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",  # Penicillin core
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (duplicate for testing)
]


def get_common_molecules_batch() -> Dict[str, Any]:
    """Returns a batch with common test molecules."""
    return create_batch_with_specific_molecules(COMMON_SMILES[:5])


def get_large_molecule_batch() -> Dict[str, Any]:
    """Returns a batch with larger molecules for testing."""
    large_smiles = [
        "CC1=C(C2=C(O1)C=C(C=C2)OC3=NC=NC4=CC(=C(C=C43)OC)OC)C(=O)NC",  # Fruquintinib
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # Phenylalanine
        "C1=CC=C(C=C1)CC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",  # Penicillin core
    ]
    return create_batch_with_specific_molecules(large_smiles)


# Edge cases for testing
def get_edge_case_molecules() -> List[Dict[str, Any]]:
    """Returns molecules that test edge cases."""
    edge_cases = []

    # Very small molecule
    edge_cases.append(create_molecule_with_properties("C", {"molecular_weight": 12.01}))

    # Linear chain
    edge_cases.append(
        create_molecule_with_properties("CCCCCCCCCC", {"molecular_weight": 142.28})
    )

    # Ring system
    edge_cases.append(
        create_molecule_with_properties(
            "c1ccc2c(c1)ccc1ccccc21", {"molecular_weight": 178.23}
        )
    )

    # Heteroatoms
    edge_cases.append(
        create_molecule_with_properties("c1ccc2c(c1)nsn2", {"molecular_weight": 162.22})
    )

    # Charged species
    edge_cases.append(
        create_molecule_with_properties("C[NH3+]", {"molecular_weight": 32.06})
    )

    return edge_cases


def get_invalid_molecules() -> List[Dict[str, Any]]:
    """Returns invalid molecular data for testing error handling."""
    return [
        {
            "smiles": "invalid_smiles",
            "atoms": np.array([], dtype=np.uint8),
            "coords": np.array([], dtype=np.float32).reshape(0, 3),
            "source_collection": "invalid",
            "mod_molecule": 999,
        },
        {
            "smiles": "",
            "atoms": np.array([], dtype=np.uint8),
            "coords": np.array([], dtype=np.float32).reshape(0, 3),
            "source_collection": "empty",
            "mod_molecule": 998,
        },
    ]


# E3GNN-specific fixtures for PyTorch tensors
def get_e3gnn_compatible_batch(
    batch_size: int = 3, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Returns batch data compatible with e3gnn_clip model.

    Args:
        batch_size: Number of molecules in batch
        device: Device to place tensors on

    Returns:
        Dictionary with 'atoms' and 'coords' as PyTorch tensors
    """
    batch_data = get_batch_data(batch_size)

    # Convert to PyTorch tensors
    atoms = torch.tensor(batch_data["atoms"], dtype=torch.long, device=device)
    coords = torch.tensor(batch_data["coords"], dtype=torch.float32, device=device)

    return {"atoms": atoms, "coords": coords}


def get_e3gnn_single_molecule(device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Returns single molecule data compatible with e3gnn_clip model.

    Args:
        device: Device to place tensors on

    Returns:
        Dictionary with 'atoms' and 'coords' as PyTorch tensors
    """
    mol_data = get_simple_molecule_data()

    # Add batch dimension and convert to tensors
    atoms = torch.tensor(mol_data["atoms"], dtype=torch.long, device=device).unsqueeze(
        0
    )
    coords = torch.tensor(
        mol_data["coords"], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return {"atoms": atoms, "coords": coords}


def get_e3gnn_batch_with_padding(
    batch_size: int = 4, max_atoms: int = 20, device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Returns batch data with explicit padding for e3gnn_clip testing.

    Args:
        batch_size: Number of molecules in batch
        max_atoms: Maximum number of atoms (for padding)
        device: Device to place tensors on

    Returns:
        Dictionary with padded 'atoms' and 'coords' as PyTorch tensors
    """
    # Create molecules of different sizes
    molecules = [
        get_simple_molecule_data(),  # 6 atoms
        get_ethanol_data(),  # 3 atoms
        get_aspirin_data(),  # 16 atoms
        get_simple_molecule_data(),  # 6 atoms
    ]

    # Pad to batch_size if needed
    while len(molecules) < batch_size:
        molecules.append(get_simple_molecule_data())

    # Initialize tensors with padding
    atoms = torch.zeros((batch_size, max_atoms), dtype=torch.long, device=device)
    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32, device=device)

    for i, mol in enumerate(molecules):
        n_atoms = len(mol["atoms"])
        if n_atoms <= max_atoms:
            atoms[i, :n_atoms] = torch.tensor(mol["atoms"], dtype=torch.long)
            coords[i, :n_atoms, :] = torch.tensor(mol["coords"], dtype=torch.float32)
        else:
            # Truncate if molecule is too large
            atoms[i, :max_atoms] = torch.tensor(
                mol["atoms"][:max_atoms], dtype=torch.long
            )
            coords[i, :max_atoms, :] = torch.tensor(
                mol["coords"][:max_atoms], dtype=torch.float32
            )

    return {"atoms": atoms, "coords": coords}


def get_e3gnn_edge_cases(device: str = "cpu") -> List[Dict[str, torch.Tensor]]:
    """
    Returns edge case molecules for e3gnn_clip testing.

    Args:
        device: Device to place tensors on

    Returns:
        List of dictionaries with 'atoms' and 'coords' as PyTorch tensors
    """
    edge_cases = []

    # Single atom
    single_atom = {
        "atoms": torch.tensor([[6]], dtype=torch.long, device=device),  # Carbon
        "coords": torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
    }
    edge_cases.append(single_atom)

    # Two atoms
    two_atoms = {
        "atoms": torch.tensor([[6, 8]], dtype=torch.long, device=device),  # C-O
        "coords": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]], dtype=torch.float32, device=device
        ),
    }
    edge_cases.append(two_atoms)

    # Linear chain (10 carbons)
    linear_chain = {
        "atoms": torch.tensor(
            [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6]], dtype=torch.long, device=device
        ),
        "coords": torch.tensor(
            [[[i * 1.5, 0.0, 0.0] for i in range(10)]],
            dtype=torch.float32,
            device=device,
        ),
    }
    edge_cases.append(linear_chain)

    # Ring with heteroatoms
    ring_hetero = {
        "atoms": torch.tensor(
            [[6, 6, 7, 6, 6, 6]], dtype=torch.long, device=device
        ),  # Pyridine-like
        "coords": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                    [2.1, 1.2, 0.0],
                    [1.4, 2.4, 0.0],
                    [0.0, 2.4, 0.0],
                    [-0.7, 1.2, 0.0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        ),
    }
    edge_cases.append(ring_hetero)

    return edge_cases


def get_e3gnn_invalid_inputs(device: str = "cpu") -> List[Dict[str, torch.Tensor]]:
    """
    Returns invalid inputs for testing e3gnn_clip error handling.

    Args:
        device: Device to place tensors on

    Returns:
        List of dictionaries with invalid 'atoms' and 'coords' as PyTorch tensors
    """
    invalid_inputs = []

    # Empty molecule
    empty_mol = {
        "atoms": torch.tensor([[]], dtype=torch.long, device=device),
        "coords": torch.tensor([[]], dtype=torch.float32, device=device).reshape(
            1, 0, 3
        ),
    }
    invalid_inputs.append(empty_mol)

    # Invalid atomic numbers (>84)
    invalid_atoms = {
        "atoms": torch.tensor([[100, 101, 102]], dtype=torch.long, device=device),
        "coords": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        ),
    }
    invalid_inputs.append(invalid_atoms)

    # Mismatched shapes
    mismatched_shapes = {
        "atoms": torch.tensor([[6, 6, 6]], dtype=torch.long, device=device),
        "coords": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32, device=device
        ),  # Only 2 coords for 3 atoms
    }
    invalid_inputs.append(mismatched_shapes)

    return invalid_inputs


def convert_batch_to_e3gnn_format(
    batch_data: Dict[str, Any], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Converts any batch data to e3gnn_clip compatible format.

    Args:
        batch_data: Dictionary with 'atoms' and 'coords' as numpy arrays
        device: Device to place tensors on

    Returns:
        Dictionary with 'atoms' and 'coords' as PyTorch tensors
    """
    atoms = torch.tensor(batch_data["atoms"], dtype=torch.long, device=device)
    coords = torch.tensor(batch_data["coords"], dtype=torch.float32, device=device)

    return {"atoms": atoms, "coords": coords}


def get_e3gnn_test_suite(device: str = "cpu") -> Dict[str, Any]:
    """
    Returns a comprehensive test suite for e3gnn_clip model.

    Args:
        device: Device to place tensors on

    Returns:
        Dictionary containing various test cases
    """
    return {
        "basic_batch": get_e3gnn_compatible_batch(3, device),
        "single_molecule": get_e3gnn_single_molecule(device),
        "padded_batch": get_e3gnn_batch_with_padding(4, 20, device),
        "edge_cases": get_e3gnn_edge_cases(device),
        "invalid_inputs": get_e3gnn_invalid_inputs(device),
        "large_molecules": get_e3gnn_compatible_batch(
            2, device
        ),  # Will include aspirin
        "small_molecules": get_e3gnn_compatible_batch(
            5, device
        ),  # Will include ethanol and benzene
    }


# Example usage function
def test_e3gnn_with_fixtures():
    """
    Example function showing how to use fixtures with e3gnn_clip model.
    """
    try:
        from coati.models.encoding.e3gnn_clip import e3gnn_clip

        # Get test data
        test_data = get_e3gnn_test_suite("cpu")

        # Initialize model
        model = e3gnn_clip(hidden_nf=128, device="cpu", n_layers=3, message_cutoff=5)

        # Test basic batch
        batch = test_data["basic_batch"]
        output = model(batch["atoms"], batch["coords"])
        print(f"Basic batch output shape: {output.shape}")

        # Test single molecule
        single = test_data["single_molecule"]
        output = model(single["atoms"], single["coords"])
        print(f"Single molecule output shape: {output.shape}")

        # Test edge cases
        for i, edge_case in enumerate(test_data["edge_cases"]):
            try:
                output = model(edge_case["atoms"], edge_case["coords"])
                print(f"Edge case {i} output shape: {output.shape}")
            except Exception as e:
                print(f"Edge case {i} failed: {e}")

        print("E3GNN fixture testing completed successfully!")

    except ImportError:
        print("e3gnn_clip model not available for testing")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    # Run the test if this file is executed directly
    test_e3gnn_with_fixtures()
