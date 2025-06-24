"""
Test equivariance properties of e3gnn_clip model.

This module tests that the e3gnn_clip model is equivariant to:
1. Rotations of the input coordinates
2. Translations of the input coordinates
3. Permutations of atoms within molecules
4. Combinations of the above transformations
"""

import torch
import numpy as np
import pytest
from typing import Tuple, List
import math

from coati.test.fixtures import get_e3gnn_compatible_batch, get_e3gnn_single_molecule
from coati.models.encoding.e3gnn_clip import e3gnn_clip


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
    # Reshape for matrix multiplication: (batch_size, n_atoms, 3) @ (3, 3)
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


def test_rotation_equivariance():
    """Test that the model is equivariant to rotations."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Get original output
    with torch.no_grad():
        original_output = model(atoms, coords)

    # Test multiple rotation angles
    angles = [0.5, 1.0, 2.0, 3.0]
    axes = ["x", "y", "z"]

    for angle in angles:
        for axis in axes:
            # Create rotation matrix
            rotation_matrix = create_rotation_matrix(angle, axis)

            # Apply rotation to coordinates
            rotated_coords = apply_rotation(coords, rotation_matrix)

            # Get output for rotated coordinates
            with torch.no_grad():
                rotated_output = model(atoms, rotated_coords)

            # Check that outputs are approximately equal (equivariance)
            # Note: For exact equivariance, we'd need to apply the same rotation to the output
            # But since we're testing the model's behavior, we check that the outputs are close
            # This is a simplified test - in practice, you might want to check specific properties
            assert torch.allclose(
                original_output, rotated_output, atol=1e-5, rtol=1e-5
            ), f"Model output changed under {axis}-rotation by {angle} radians"


def test_translation_equivariance():
    """Test that the model is equivariant to translations."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Get original output
    with torch.no_grad():
        original_output = model(atoms, coords)

    # Test multiple translations
    translations = [
        torch.tensor([1.0, 0.0, 0.0]),  # Translate along x
        torch.tensor([0.0, 1.0, 0.0]),  # Translate along y
        torch.tensor([0.0, 0.0, 1.0]),  # Translate along z
        torch.tensor([1.0, 1.0, 1.0]),  # Translate along diagonal
        torch.tensor([-2.0, 3.0, -1.5]),  # Random translation
    ]

    for translation in translations:
        # Apply translation to coordinates
        translated_coords = apply_translation(coords, translation)

        # Get output for translated coordinates
        with torch.no_grad():
            translated_output = model(atoms, translated_coords)

        # Check that outputs are exactly equal (translation equivariance)
        assert torch.allclose(
            original_output, translated_output, atol=1e-6, rtol=1e-6
        ), f"Model output changed under translation {translation}"


def test_permutation_equivariance():
    """Test that the model is equivariant to atom permutations."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Get test data (use single molecule for easier permutation testing)
    mol_data = get_e3gnn_single_molecule(device)
    atoms = mol_data["atoms"]
    coords = mol_data["coords"]

    # Get original output
    with torch.no_grad():
        original_output = model(atoms, coords)

    # Test multiple permutations
    n_atoms = atoms.shape[1]
    permutations = [
        list(range(n_atoms)),  # Identity permutation
        list(range(n_atoms - 1, -1, -1)),  # Reverse permutation
        [1, 0, 2, 3, 4, 5] if n_atoms >= 6 else list(range(n_atoms)),  # Swap first two
        (
            [2, 1, 0, 3, 4, 5] if n_atoms >= 6 else list(range(n_atoms))
        ),  # Rotate first three
    ]

    for permutation in permutations:
        if len(permutation) <= n_atoms:
            # Apply permutation
            permuted_atoms, permuted_coords = apply_permutation(
                atoms, coords, permutation
            )

            # Get output for permuted atoms/coords
            with torch.no_grad():
                permuted_output = model(permuted_atoms, permuted_coords)

            # Check that outputs are exactly equal (permutation equivariance)
            assert torch.allclose(
                original_output, permuted_output, atol=1e-6, rtol=1e-6
            ), f"Model output changed under permutation {permutation}"


def test_combined_transformations():
    """Test that the model is equivariant to combinations of transformations."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Get original output
    with torch.no_grad():
        original_output = model(atoms, coords)

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

    # Get output for transformed coordinates
    with torch.no_grad():
        transformed_output = model(atoms_transformed, coords_transformed)

    # Check that outputs are approximately equal
    # Note: For exact equivariance under rotation, we'd need to apply the inverse rotation to the output
    assert torch.allclose(
        original_output, transformed_output, atol=1e-5, rtol=1e-5
    ), "Model output changed under combined transformations"


def test_edge_cases():
    """Test equivariance with edge cases."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Test with single atom
    single_atom_data = {
        "atoms": torch.tensor([[6]], dtype=torch.long, device=device),
        "coords": torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        original_output = model(single_atom_data["atoms"], single_atom_data["coords"])

    # Apply translation
    translation = torch.tensor([10.0, 20.0, 30.0])
    translated_coords = apply_translation(single_atom_data["coords"], translation)

    with torch.no_grad():
        translated_output = model(single_atom_data["atoms"], translated_coords)

    # Should be exactly equal for single atom
    assert torch.allclose(
        original_output, translated_output, atol=1e-6, rtol=1e-6
    ), "Single atom output changed under translation"


def test_gradient_equivariance():
    """Test that gradients are also equivariant."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Get test data
    batch_data = get_e3gnn_compatible_batch(2, device)
    atoms = batch_data["atoms"]
    coords = batch_data["coords"]

    # Enable gradients
    coords.requires_grad_(True)

    # Get original output and gradients
    original_output = model(atoms, coords)
    original_loss = original_output.sum()
    original_loss.backward()
    original_grads = coords.grad.clone()

    # Reset gradients
    coords.grad.zero_()

    # Apply rotation
    rotation_matrix = create_rotation_matrix(1.0, "z")
    rotated_coords = apply_rotation(coords, rotation_matrix)

    # Get output and gradients for rotated coordinates
    rotated_output = model(atoms, rotated_coords)
    rotated_loss = rotated_output.sum()
    rotated_loss.backward()
    rotated_grads = coords.grad.clone()

    # Check that gradients are related by the same transformation
    # The gradients should be rotated by the inverse of the rotation matrix
    expected_rotated_grads = torch.matmul(original_grads, rotation_matrix)

    assert torch.allclose(
        rotated_grads, expected_rotated_grads, atol=1e-5, rtol=1e-5
    ), "Gradients are not equivariant under rotation"


def test_batch_equivariance():
    """Test that equivariance holds across different batch sizes."""
    device = "cpu"

    # Initialize model
    model = e3gnn_clip(hidden_nf=128, device=device, n_layers=3, message_cutoff=5)

    # Test different batch sizes
    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:
        # Get test data
        batch_data = get_e3gnn_compatible_batch(batch_size, device)
        atoms = batch_data["atoms"]
        coords = batch_data["coords"]

        # Get original output
        with torch.no_grad():
            original_output = model(atoms, coords)

        # Apply translation
        translation = torch.tensor([1.0, 1.0, 1.0])
        translated_coords = apply_translation(coords, translation)

        # Get output for translated coordinates
        with torch.no_grad():
            translated_output = model(atoms, translated_coords)

        # Check equivariance
        assert torch.allclose(
            original_output, translated_output, atol=1e-6, rtol=1e-6
        ), f"Batch size {batch_size}: Model output changed under translation"


def test_model_configurations():
    """Test equivariance with different model configurations."""
    device = "cpu"

    # Test different configurations
    configs = [
        {"hidden_nf": 64, "n_layers": 2, "message_cutoff": 3},
        {"hidden_nf": 256, "n_layers": 5, "message_cutoff": 10},
        {"hidden_nf": 128, "n_layers": 3, "message_cutoff": 5, "torch_emb": True},
        {"hidden_nf": 128, "n_layers": 3, "message_cutoff": 5, "residual": True},
    ]

    for config in configs:
        # Initialize model with current config
        model = e3gnn_clip(device=device, **config)

        # Get test data
        batch_data = get_e3gnn_compatible_batch(2, device)
        atoms = batch_data["atoms"]
        coords = batch_data["coords"]

        # Get original output
        with torch.no_grad():
            original_output = model(atoms, coords)

        # Apply translation
        translation = torch.tensor([1.0, 1.0, 1.0])
        translated_coords = apply_translation(coords, translation)

        # Get output for translated coordinates
        with torch.no_grad():
            translated_output = model(atoms, translated_coords)

        # Check equivariance
        assert torch.allclose(
            original_output, translated_output, atol=1e-6, rtol=1e-6
        ), f"Config {config}: Model output changed under translation"


if __name__ == "__main__":
    # Run all tests
    print("Testing rotation equivariance...")
    test_rotation_equivariance()

    print("Testing translation equivariance...")
    test_translation_equivariance()

    print("Testing permutation equivariance...")
    test_permutation_equivariance()

    print("Testing combined transformations...")
    test_combined_transformations()

    print("Testing edge cases...")
    test_edge_cases()

    print("Testing gradient equivariance...")
    test_gradient_equivariance()

    print("Testing batch equivariance...")
    test_batch_equivariance()

    print("Testing model configurations...")
    test_model_configurations()

    print("All equivariance tests passed!")
