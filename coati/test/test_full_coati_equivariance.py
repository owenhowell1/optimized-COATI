"""
Test equivariance properties of the full COATI model (e3gnn_smiles_clip_e2e).

This module tests that the full COATI model is equivariant to:
1. Rotations of the input coordinates (3D encoder should be invariant)
2. Translations of the input coordinates (3D encoder should be invariant)
3. Permutations of atoms within molecules (3D encoder should be invariant)
4. Combinations of the above transformations
5. Full model behavior under these transformations

The full COATI model combines:
- E3GNN encoder for 3D molecular representations
- Transformer encoder for SMILES sequences
- CLIP-style contrastive learning between modalities
"""

import torch
import numpy as np
import pytest
from typing import Tuple, List, Dict, Any
import math

from coati.test.fixtures import (
    get_e3gnn_compatible_batch,
    get_e3gnn_single_molecule,
    get_e3gnn_batch_with_padding,
    get_e3gnn_edge_cases,
    get_e3gnn_invalid_inputs,
    get_e3gnn_test_suite,
)
from coati.models.encoding.clip_e2e import e3gnn_smiles_clip_e2e
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from coati.models.encoding.tokenizers import get_vocab


def create_rotation_matrix(angle: float, axis: str = "z") -> torch.Tensor:
    """
    Create a 3x3 rotation matrix for rotation around specified axis.

    Args:
        angle: Rotation angle in radians
        axis: Axis of rotation ('x', 'y', or 'z')

    Returns:
        3x3 rotation matrix as torch.Tensor
    """
    cos_a = torch.cos(torch.tensor(angle))
    sin_a = torch.sin(torch.tensor(angle))

    if axis == "x":
        return torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]]
        )
    elif axis == "y":
        return torch.tensor(
            [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
        )
    elif axis == "z":
        return torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]]
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")


def create_random_rotation_matrix() -> torch.Tensor:
    """
    Create a random 3x3 rotation matrix using Euler angles.

    Returns:
        3x3 rotation matrix as torch.Tensor
    """
    # Random Euler angles
    alpha = torch.rand(1) * 2 * math.pi  # Rotation around z
    beta = torch.rand(1) * 2 * math.pi  # Rotation around y
    gamma = torch.rand(1) * 2 * math.pi  # Rotation around z

    # Rotation matrices
    Rz1 = create_rotation_matrix(alpha, "z")
    Ry = create_rotation_matrix(beta, "y")
    Rz2 = create_rotation_matrix(gamma, "z")

    # Combined rotation
    return Rz2 @ Ry @ Rz1


def apply_rotation(coords: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation to coordinates.

    Args:
        coords: Coordinates tensor of shape (batch_size, n_atoms, 3)
        rotation_matrix: 3x3 rotation matrix

    Returns:
        Rotated coordinates
    """
    return torch.matmul(coords, rotation_matrix.T)


def apply_translation(coords: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Apply translation to coordinates.

    Args:
        coords: Coordinates tensor of shape (batch_size, n_atoms, 3)
        translation: Translation vector of shape (3,)

    Returns:
        Translated coordinates
    """
    return coords + translation.unsqueeze(0).unsqueeze(0)


def apply_permutation(
    atoms: torch.Tensor, coords: torch.Tensor, permutation: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply permutation to atoms and coordinates.

    Args:
        atoms: Atomic numbers tensor of shape (batch_size, n_atoms)
        coords: Coordinates tensor of shape (batch_size, n_atoms, 3)
        permutation: List of indices defining the permutation

    Returns:
        Tuple of (permuted_atoms, permuted_coords)
    """
    return atoms[:, permutation], coords[:, permutation, :]


def create_mock_tokenizer(device: str = "cpu") -> TrieTokenizer:
    """
    Create a mock tokenizer for testing.

    Args:
        device: Device to place tokenizer on

    Returns:
        Mock tokenizer
    """
    # Use a simple vocabulary for testing
    vocab = get_vocab("coati2_12_12.json")
    tokenizer = TrieTokenizer(n_seq=50, **vocab)
    return tokenizer


def create_mock_tokens(
    batch_size: int, seq_len: int = 20, device: str = "cpu"
) -> torch.Tensor:
    """
    Create mock token sequences for testing.

    Args:
        batch_size: Number of sequences
        seq_len: Length of each sequence
        device: Device to place tokens on

    Returns:
        Token tensor of shape (batch_size, seq_len)
    """
    # Create mock SMILES-like tokens
    tokens = torch.randint(
        1, 100, (batch_size, seq_len), dtype=torch.long, device=device
    )
    # Ensure no padding tokens at the beginning
    tokens[:, 0] = 1  # Start token
    tokens[:, -1] = 2  # End token
    return tokens


def test_e3gnn_encoder_equivariance():
    """Test that the E3GNN encoder component is equivariant to coordinate transformations."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Test E3GNN encoder directly
    with torch.no_grad():
        original_encoding = model.encode_points(atoms, coords)

    # Test translation invariance
    translation = torch.tensor([1.0, 1.0, 1.0])
    translated_coords = apply_translation(coords, translation)

    with torch.no_grad():
        translated_encoding = model.encode_points(atoms, translated_coords)

    # Should be exactly equal (translation invariance)
    assert torch.allclose(
        original_encoding, translated_encoding, atol=1e-6, rtol=1e-6
    ), "E3GNN encoder output changed under translation"

    # Test rotation invariance (with tolerance for numerical precision)
    rotation_matrix = create_rotation_matrix(1.0, "z")
    rotated_coords = apply_rotation(coords, rotation_matrix)

    with torch.no_grad():
        rotated_encoding = model.encode_points(atoms, rotated_coords)

    # Should be approximately equal (rotation invariance)
    assert torch.allclose(
        original_encoding, rotated_encoding, atol=1e-5, rtol=1e-5
    ), "E3GNN encoder output changed under rotation"

    # Test permutation invariance
    n_atoms = atoms.shape[1]
    if n_atoms >= 4:
        permutation = [1, 0, 3, 2] + list(range(4, n_atoms))
        permuted_atoms, permuted_coords = apply_permutation(atoms, coords, permutation)

        with torch.no_grad():
            permuted_encoding = model.encode_points(permuted_atoms, permuted_coords)

        # Should be exactly equal (permutation invariance)
        assert torch.allclose(
            original_encoding, permuted_encoding, atol=1e-6, rtol=1e-6
        ), "E3GNN encoder output changed under permutation"


def test_full_model_forward_equivariance():
    """Test that the full model's forward pass is equivariant to coordinate transformations."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Create mock tokens
    raw_tokens = create_mock_tokens(2, 20, device)
    augmented_tokens = create_mock_tokens(2, 20, device)

    # Get original forward pass results
    with torch.no_grad():
        h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
            raw_tokens, augmented_tokens, atoms, coords, tokenizer
        )

    # Test translation invariance
    translation = torch.tensor([1.0, 1.0, 1.0])
    translated_coords = apply_translation(coords, translation)

    with torch.no_grad():
        h_e3gnn_trans, h_smiles_trans, logits_trans, clip_loss_trans = model.forward(
            raw_tokens, augmented_tokens, atoms, translated_coords, tokenizer
        )

    # E3GNN embeddings should be exactly equal
    assert torch.allclose(
        h_e3gnn_orig, h_e3gnn_trans, atol=1e-6, rtol=1e-6
    ), "E3GNN embeddings changed under translation"

    # SMILES embeddings should be exactly equal (same tokens)
    assert torch.allclose(
        h_smiles_orig, h_smiles_trans, atol=1e-6, rtol=1e-6
    ), "SMILES embeddings changed under translation"

    # Logits should be exactly equal (same input to transformer)
    assert torch.allclose(
        logits_orig, logits_trans, atol=1e-6, rtol=1e-6
    ), "Transformer logits changed under translation"

    # CLIP loss should be exactly equal
    assert torch.allclose(
        clip_loss_orig, clip_loss_trans, atol=1e-6, rtol=1e-6
    ), "CLIP loss changed under translation"


def test_full_model_rotation_equivariance():
    """Test that the full model is equivariant to rotations."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Create mock tokens
    raw_tokens = create_mock_tokens(2, 20, device)
    augmented_tokens = create_mock_tokens(2, 20, device)

    # Get original forward pass results
    with torch.no_grad():
        h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
            raw_tokens, augmented_tokens, atoms, coords, tokenizer
        )

    # Test multiple rotation angles
    angles = [0.5, 1.0, 2.0]
    axes = ["x", "y", "z"]

    for angle in angles:
        for axis in axes:
            # Create rotation matrix
            rotation_matrix = create_rotation_matrix(angle, axis)

            # Apply rotation to coordinates
            rotated_coords = apply_rotation(coords, rotation_matrix)

            # Get forward pass results for rotated coordinates
            with torch.no_grad():
                h_e3gnn_rot, h_smiles_rot, logits_rot, clip_loss_rot = model.forward(
                    raw_tokens, augmented_tokens, atoms, rotated_coords, tokenizer
                )

            # E3GNN embeddings should be approximately equal (rotation invariance)
            assert torch.allclose(
                h_e3gnn_orig, h_e3gnn_rot, atol=1e-5, rtol=1e-5
            ), f"E3GNN embeddings changed under {axis}-rotation by {angle} radians"

            # SMILES embeddings should be exactly equal (same tokens)
            assert torch.allclose(
                h_smiles_orig, h_smiles_rot, atol=1e-6, rtol=1e-6
            ), f"SMILES embeddings changed under {axis}-rotation by {angle} radians"

            # Logits should be exactly equal (same input to transformer)
            assert torch.allclose(
                logits_orig, logits_rot, atol=1e-6, rtol=1e-6
            ), f"Transformer logits changed under {axis}-rotation by {angle} radians"


def test_full_model_permutation_equivariance():
    """Test that the full model is equivariant to atom permutations."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Get test data (use single molecule for easier permutation testing)
    mol_data = get_e3gnn_single_molecule(device)
    atoms = mol_data["atoms"]
    coords = mol_data["coords"]

    # Create mock tokens
    raw_tokens = create_mock_tokens(1, 20, device)
    augmented_tokens = create_mock_tokens(1, 20, device)

    # Get original forward pass results
    with torch.no_grad():
        h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
            raw_tokens, augmented_tokens, atoms, coords, tokenizer
        )

    # Test multiple permutations
    n_atoms = atoms.shape[1]
    permutations = [
        list(range(n_atoms)),  # Identity permutation
        list(range(n_atoms - 1, -1, -1)),  # Reverse permutation
        [1, 0, 2, 3, 4, 5] if n_atoms >= 6 else list(range(n_atoms)),  # Swap first two
    ]

    for permutation in permutations:
        if len(permutation) <= n_atoms:
            # Apply permutation
            permuted_atoms, permuted_coords = apply_permutation(
                atoms, coords, permutation
            )

            # Get forward pass results for permuted atoms/coords
            with torch.no_grad():
                h_e3gnn_perm, h_smiles_perm, logits_perm, clip_loss_perm = (
                    model.forward(
                        raw_tokens,
                        augmented_tokens,
                        permuted_atoms,
                        permuted_coords,
                        tokenizer,
                    )
                )

            # E3GNN embeddings should be exactly equal (permutation invariance)
            assert torch.allclose(
                h_e3gnn_orig, h_e3gnn_perm, atol=1e-6, rtol=1e-6
            ), f"E3GNN embeddings changed under permutation {permutation}"

            # SMILES embeddings should be exactly equal (same tokens)
            assert torch.allclose(
                h_smiles_orig, h_smiles_perm, atol=1e-6, rtol=1e-6
            ), f"SMILES embeddings changed under permutation {permutation}"

            # Logits should be exactly equal (same input to transformer)
            assert torch.allclose(
                logits_orig, logits_perm, atol=1e-6, rtol=1e-6
            ), f"Transformer logits changed under permutation {permutation}"


def test_full_model_combined_transformations():
    """Test that the full model is equivariant to combinations of transformations."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Create mock tokens
    raw_tokens = create_mock_tokens(2, 20, device)
    augmented_tokens = create_mock_tokens(2, 20, device)

    # Get original forward pass results
    with torch.no_grad():
        h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
            raw_tokens, augmented_tokens, atoms, coords, tokenizer
        )

    # Apply combination of transformations
    # 1. Random rotation
    rotation_matrix = create_random_rotation_matrix()
    coords_transformed = apply_rotation(coords, rotation_matrix)

    # 2. Translation
    translation = torch.tensor([1.5, -2.0, 0.8])
    coords_transformed = apply_translation(coords_transformed, translation)

    # 3. Permutation (if we have enough atoms)
    n_atoms = atoms.shape[1]
    if n_atoms >= 4:
        permutation = [1, 0, 3, 2] + list(range(4, n_atoms))
        atoms_transformed, coords_transformed = apply_permutation(
            atoms, coords_transformed, permutation
        )
    else:
        atoms_transformed = atoms

    # Get forward pass results for transformed coordinates
    with torch.no_grad():
        h_e3gnn_trans, h_smiles_trans, logits_trans, clip_loss_trans = model.forward(
            raw_tokens,
            augmented_tokens,
            atoms_transformed,
            coords_transformed,
            tokenizer,
        )

    # E3GNN embeddings should be approximately equal (invariant to combined transformations)
    assert torch.allclose(
        h_e3gnn_orig, h_e3gnn_trans, atol=1e-5, rtol=1e-5
    ), "E3GNN embeddings changed under combined transformations"

    # SMILES embeddings should be exactly equal (same tokens)
    assert torch.allclose(
        h_smiles_orig, h_smiles_trans, atol=1e-6, rtol=1e-6
    ), "SMILES embeddings changed under combined transformations"

    # Logits should be exactly equal (same input to transformer)
    assert torch.allclose(
        logits_orig, logits_trans, atol=1e-6, rtol=1e-6
    ), "Transformer logits changed under combined transformations"


def test_full_model_edge_cases():
    """Test the full model with edge cases."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Test with single atom
    single_atom_data = {
        "atoms": torch.tensor([[6]], dtype=torch.long, device=device),
        "coords": torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
    }

    raw_tokens = create_mock_tokens(1, 20, device)
    augmented_tokens = create_mock_tokens(1, 20, device)

    with torch.no_grad():
        h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
            raw_tokens,
            augmented_tokens,
            single_atom_data["atoms"],
            single_atom_data["coords"],
            tokenizer,
        )

    # Apply translation
    translation = torch.tensor([10.0, 20.0, 30.0])
    translated_coords = apply_translation(single_atom_data["coords"], translation)

    with torch.no_grad():
        h_e3gnn_trans, h_smiles_trans, logits_trans, clip_loss_trans = model.forward(
            raw_tokens,
            augmented_tokens,
            single_atom_data["atoms"],
            translated_coords,
            tokenizer,
        )

    # Should be exactly equal for single atom
    assert torch.allclose(
        h_e3gnn_orig, h_e3gnn_trans, atol=1e-6, rtol=1e-6
    ), "Single atom E3GNN embeddings changed under translation"


def test_full_model_gradient_equivariance():
    """Test that gradients are also equivariant."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Enable gradients for coordinates
    coords.requires_grad_(True)

    # Create mock tokens
    raw_tokens = create_mock_tokens(2, 20, device)
    augmented_tokens = create_mock_tokens(2, 20, device)

    # Get original output and gradients
    h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
        raw_tokens, augmented_tokens, atoms, coords, tokenizer
    )
    total_loss_orig = clip_loss_orig.sum()
    total_loss_orig.backward()
    original_grads = coords.grad.clone()

    # Reset gradients
    coords.grad.zero_()

    # Apply rotation
    rotation_matrix = create_rotation_matrix(1.0, "z")
    rotated_coords = apply_rotation(coords, rotation_matrix)

    # Get output and gradients for rotated coordinates
    h_e3gnn_rot, h_smiles_rot, logits_rot, clip_loss_rot = model.forward(
        raw_tokens, augmented_tokens, atoms, rotated_coords, tokenizer
    )
    total_loss_rot = clip_loss_rot.sum()
    total_loss_rot.backward()
    rotated_grads = coords.grad.clone()

    # Check that gradients are related by the same transformation
    # The gradients should be rotated by the inverse of the rotation matrix
    expected_rotated_grads = torch.matmul(original_grads, rotation_matrix)

    assert torch.allclose(
        rotated_grads, expected_rotated_grads, atol=1e-5, rtol=1e-5
    ), "Gradients are not equivariant under rotation"


def test_full_model_configurations():
    """Test equivariance with different model configurations."""
    device = "cpu"

    # Test different configurations
    configs = [
        {
            "n_layer_e3gnn": 2,
            "n_layer_xformer": 2,
            "n_hidden_xformer": 32,
            "n_hidden_e3nn": 32,
            "msg_cutoff_e3nn": 3.0,
            "n_embd_common": 32,
            "n_head": 2,
            "n_seq": 30,
            "use_point_encoder": True,
        },
        {
            "n_layer_e3gnn": 4,
            "n_layer_xformer": 6,
            "n_hidden_xformer": 128,
            "n_hidden_e3nn": 128,
            "msg_cutoff_e3nn": 8.0,
            "n_embd_common": 128,
            "n_head": 8,
            "n_seq": 100,
            "use_point_encoder": True,
        },
        {
            "n_layer_e3gnn": 3,
            "n_layer_xformer": 4,
            "n_hidden_xformer": 64,
            "n_hidden_e3nn": 64,
            "msg_cutoff_e3nn": 5.0,
            "n_embd_common": 64,
            "n_head": 4,
            "n_seq": 50,
            "use_point_encoder": False,  # Test without point encoder
        },
    ]

    for i, config in enumerate(configs):
        print(f"Testing configuration {i+1}/{len(configs)}")

        # Initialize model with current config
        model = e3gnn_smiles_clip_e2e(device=device, **config)

        # Create mock tokenizer
        tokenizer = create_mock_tokenizer(device)

        # Get test data
        batch_data = get_e3gnn_compatible_batch(2, device)
        atoms = batch_data["atoms"]
        coords = batch_data["coords"]

        # Create mock tokens
        raw_tokens = create_mock_tokens(2, 20, device)
        augmented_tokens = create_mock_tokens(2, 20, device)

        # Get original forward pass results
        with torch.no_grad():
            h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
                raw_tokens, augmented_tokens, atoms, coords, tokenizer
            )

        # Apply translation
        translation = torch.tensor([1.0, 1.0, 1.0])
        translated_coords = apply_translation(coords, translation)

        # Get forward pass results for translated coordinates
        with torch.no_grad():
            h_e3gnn_trans, h_smiles_trans, logits_trans, clip_loss_trans = (
                model.forward(
                    raw_tokens, augmented_tokens, atoms, translated_coords, tokenizer
                )
            )

        # Check equivariance
        if config["use_point_encoder"]:
            # E3GNN embeddings should be exactly equal (translation invariance)
            assert torch.allclose(
                h_e3gnn_orig, h_e3gnn_trans, atol=1e-6, rtol=1e-6
            ), f"Config {i}: E3GNN embeddings changed under translation"
        else:
            # When point encoder is disabled, E3GNN embeddings should be zero
            assert torch.allclose(
                h_e3gnn_orig, torch.zeros_like(h_e3gnn_orig), atol=1e-6, rtol=1e-6
            ), f"Config {i}: E3GNN embeddings should be zero when point encoder is disabled"

        # SMILES embeddings should be exactly equal (same tokens)
        assert torch.allclose(
            h_smiles_orig, h_smiles_trans, atol=1e-6, rtol=1e-6
        ), f"Config {i}: SMILES embeddings changed under translation"

        # Logits should be exactly equal (same input to transformer)
        assert torch.allclose(
            logits_orig, logits_trans, atol=1e-6, rtol=1e-6
        ), f"Config {i}: Transformer logits changed under translation"


def test_full_model_batch_equivariance():
    """Test that equivariance holds across different batch sizes."""
    device = "cpu"

    # Initialize the full model
    model = e3gnn_smiles_clip_e2e(
        n_layer_e3gnn=3,
        n_layer_xformer=4,
        n_hidden_xformer=64,
        n_hidden_e3nn=64,
        msg_cutoff_e3nn=5.0,
        n_embd_common=64,
        n_head=4,
        n_seq=50,
        device=device,
        use_point_encoder=True,
    )

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer(device)

    # Test different batch sizes
    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:
        # Get test data
        batch_data = get_e3gnn_compatible_batch(batch_size, device)
        atoms = batch_data["atoms"]
        coords = batch_data["coords"]

        # Create mock tokens
        raw_tokens = create_mock_tokens(batch_size, 20, device)
        augmented_tokens = create_mock_tokens(batch_size, 20, device)

        # Get original forward pass results
        with torch.no_grad():
            h_e3gnn_orig, h_smiles_orig, logits_orig, clip_loss_orig = model.forward(
                raw_tokens, augmented_tokens, atoms, coords, tokenizer
            )

        # Apply translation
        translation = torch.tensor([1.0, 1.0, 1.0])
        translated_coords = apply_translation(coords, translation)

        # Get forward pass results for translated coordinates
        with torch.no_grad():
            h_e3gnn_trans, h_smiles_trans, logits_trans, clip_loss_trans = (
                model.forward(
                    raw_tokens, augmented_tokens, atoms, translated_coords, tokenizer
                )
            )

        # Check equivariance
        assert torch.allclose(
            h_e3gnn_orig, h_e3gnn_trans, atol=1e-6, rtol=1e-6
        ), f"Batch size {batch_size}: E3GNN embeddings changed under translation"

        assert torch.allclose(
            h_smiles_orig, h_smiles_trans, atol=1e-6, rtol=1e-6
        ), f"Batch size {batch_size}: SMILES embeddings changed under translation"

        assert torch.allclose(
            logits_orig, logits_trans, atol=1e-6, rtol=1e-6
        ), f"Batch size {batch_size}: Transformer logits changed under translation"


if __name__ == "__main__":
    # Run all tests
    print("Testing E3GNN encoder equivariance...")
    test_e3gnn_encoder_equivariance()

    print("Testing full model forward equivariance...")
    test_full_model_forward_equivariance()

    print("Testing full model rotation equivariance...")
    test_full_model_rotation_equivariance()

    print("Testing full model permutation equivariance...")
    test_full_model_permutation_equivariance()

    print("Testing full model combined transformations...")
    test_full_model_combined_transformations()

    print("Testing full model edge cases...")
    test_full_model_edge_cases()

    print("Testing full model gradient equivariance...")
    test_full_model_gradient_equivariance()

    print("Testing full model configurations...")
    test_full_model_configurations()

    print("Testing full model batch equivariance...")
    test_full_model_batch_equivariance()

    print("All full COATI model equivariance tests passed!")
