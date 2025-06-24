"""
Profile the E3GNN layer performance characteristics.

This module provides comprehensive profiling for the e3gnn_clip model including:
- Forward pass timing
- Memory usage
- Throughput analysis
- Scaling behavior with different input sizes
- Performance comparison across different configurations
"""

import torch
import torch.nn as nn
import time
import psutil
import gc
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json
import os

from coati.models.encoding.e3gnn_clip import e3gnn_clip
from coati.test.fixtures import (
    get_e3gnn_compatible_batch,
    get_e3gnn_batch_with_padding,
    get_e3gnn_edge_cases,
    get_e3gnn_test_suite,
)


@dataclass
class ProfileResult:
    """Container for profiling results."""

    model_config: Dict[str, Any]
    input_size: Tuple[int, int]  # (batch_size, max_atoms)
    forward_time: float
    memory_usage: float
    throughput: float  # molecules per second
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    input_tensor_size: float = 0.0
    output_tensor_size: float = 0.0


class E3GNNProfiler:
    """Profiler for E3GNN model performance analysis."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize the profiler.

        Args:
            device: Device to run profiling on ('cpu' or 'cuda')
        """
        self.device = device
        self.results = []

        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get GPU memory usage in MB."""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            return allocated, reserved
        return 0.0, 0.0

    def clear_gpu_cache(self):
        """Clear GPU cache to ensure accurate memory measurements."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def create_model(self, config: Dict[str, Any]) -> e3gnn_clip:
        """Create an E3GNN model with given configuration."""
        return e3gnn_clip(device=self.device, **config)

    def create_test_data(
        self, batch_size: int, max_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test data with specified batch size and max atoms."""
        # Create random atomic numbers (1-84, excluding 0 for padding)
        atoms = torch.randint(
            1, 85, (batch_size, max_atoms), dtype=torch.long, device=self.device
        )

        # Create random coordinates
        coords = torch.randn(
            batch_size, max_atoms, 3, dtype=torch.float32, device=self.device
        )

        # Apply some padding (set some atoms to 0)
        if max_atoms > 10:
            # Randomly set some atoms to 0 for padding
            for i in range(batch_size):
                n_atoms = torch.randint(5, max_atoms + 1, (1,)).item()
                atoms[i, n_atoms:] = 0
                coords[i, n_atoms:, :] = 0.0

        return atoms, coords

    def profile_single_run(
        self,
        model: e3gnn_clip,
        atoms: torch.Tensor,
        coords: torch.Tensor,
        config: Dict[str, Any],
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> ProfileResult:
        """
        Profile a single model configuration with given input data.

        Args:
            model: E3GNN model to profile
            atoms: Input atomic numbers tensor
            coords: Input coordinates tensor
            config: Model configuration
            num_warmup: Number of warmup runs
            num_runs: Number of profiling runs

        Returns:
            ProfileResult with timing and memory information
        """
        batch_size, max_atoms = atoms.shape

        # Warmup runs
        model.eval()
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(atoms, coords)

        # Clear cache before profiling
        self.clear_gpu_cache()

        # Measure memory before
        memory_before = self.get_memory_usage()
        gpu_allocated_before, gpu_reserved_before = self.get_gpu_memory_usage()

        # Profile forward pass
        model.eval()
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                output = model(atoms, coords)
                torch.cuda.synchronize() if self.device == "cuda" else None
                end_time = time.time()
                times.append(end_time - start_time)

        # Measure memory after
        memory_after = self.get_memory_usage()
        gpu_allocated_after, gpu_reserved_after = self.get_gpu_memory_usage()

        # Calculate statistics
        forward_time = np.mean(times)
        memory_usage = memory_after - memory_before
        gpu_memory_allocated = gpu_allocated_after - gpu_allocated_before
        gpu_memory_reserved = gpu_reserved_after - gpu_reserved_before

        # Calculate throughput (molecules per second)
        throughput = batch_size / forward_time

        # Calculate tensor sizes
        input_tensor_size = (
            (
                atoms.numel() * atoms.element_size()
                + coords.numel() * coords.element_size()
            )
            / 1024
            / 1024
        )  # MB
        output_tensor_size = (
            (output.numel() * output.element_size()) / 1024 / 1024
        )  # MB

        return ProfileResult(
            model_config=config,
            input_size=(batch_size, max_atoms),
            forward_time=forward_time,
            memory_usage=memory_usage,
            throughput=throughput,
            gpu_memory_allocated=gpu_memory_allocated,
            gpu_memory_reserved=gpu_memory_reserved,
            input_tensor_size=input_tensor_size,
            output_tensor_size=output_tensor_size,
        )

    def profile_configurations(
        self,
        configs: List[Dict[str, Any]],
        batch_sizes: List[int],
        max_atoms_list: List[int],
    ) -> List[ProfileResult]:
        """
        Profile multiple model configurations with different input sizes.

        Args:
            configs: List of model configurations to test
            batch_sizes: List of batch sizes to test
            max_atoms_list: List of maximum atom counts to test

        Returns:
            List of ProfileResult objects
        """
        results = []

        for config in configs:
            print(f"Profiling configuration: {config}")

            for batch_size in batch_sizes:
                for max_atoms in max_atoms_list:
                    print(f"  Batch size: {batch_size}, Max atoms: {max_atoms}")

                    # Create model
                    model = self.create_model(config)

                    # Create test data
                    atoms, coords = self.create_test_data(batch_size, max_atoms)

                    # Profile
                    try:
                        result = self.profile_single_run(model, atoms, coords, config)
                        results.append(result)
                        print(
                            f"    Forward time: {result.forward_time:.4f}s, Throughput: {result.throughput:.1f} mol/s"
                        )
                    except Exception as e:
                        print(f"    Error: {e}")

                    # Clean up
                    del model, atoms, coords
                    self.clear_gpu_cache()

        self.results.extend(results)
        return results

    def profile_scaling_behavior(
        self,
        base_config: Dict[str, Any],
        max_batch_size: int = 128,
        max_atoms: int = 50,
    ) -> List[ProfileResult]:
        """
        Profile scaling behavior with increasing batch sizes.

        Args:
            base_config: Base model configuration
            max_batch_size: Maximum batch size to test
            max_atoms: Maximum number of atoms

        Returns:
            List of ProfileResult objects
        """
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        batch_sizes = [b for b in batch_sizes if b <= max_batch_size]

        results = []

        for batch_size in batch_sizes:
            print(f"Profiling batch size: {batch_size}")

            # Create model
            model = self.create_model(base_config)

            # Create test data
            atoms, coords = self.create_test_data(batch_size, max_atoms)

            # Profile
            try:
                result = self.profile_single_run(model, atoms, coords, base_config)
                results.append(result)
                print(
                    f"  Forward time: {result.forward_time:.4f}s, Throughput: {result.throughput:.1f} mol/s"
                )
            except Exception as e:
                print(f"  Error: {e}")

            # Clean up
            del model, atoms, coords
            self.clear_gpu_cache()

        self.results.extend(results)
        return results

    def profile_memory_efficiency(
        self, config: Dict[str, Any], batch_size: int = 32, max_atoms: int = 50
    ) -> ProfileResult:
        """
        Profile memory efficiency with detailed memory analysis.

        Args:
            config: Model configuration
            batch_size: Batch size to test
            max_atoms: Maximum number of atoms

        Returns:
            ProfileResult with detailed memory information
        """
        print(
            f"Profiling memory efficiency for batch_size={batch_size}, max_atoms={max_atoms}"
        )

        # Create model
        model = self.create_model(config)

        # Create test data
        atoms, coords = self.create_test_data(batch_size, max_atoms)

        # Profile with detailed memory tracking
        result = self.profile_single_run(
            model, atoms, coords, config, num_warmup=5, num_runs=50
        )

        # Clean up
        del model, atoms, coords
        self.clear_gpu_cache()

        return result

    def generate_report(self, output_dir: str = "e3gnn_profile_results"):
        """Generate a comprehensive profiling report."""
        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        with open(os.path.join(output_dir, "profile_results.json"), "w") as f:
            json.dump(
                [
                    {
                        "model_config": r.model_config,
                        "input_size": r.input_size,
                        "forward_time": r.forward_time,
                        "memory_usage": r.memory_usage,
                        "throughput": r.throughput,
                        "gpu_memory_allocated": r.gpu_memory_allocated,
                        "gpu_memory_reserved": r.gpu_memory_reserved,
                        "input_tensor_size": r.input_tensor_size,
                        "output_tensor_size": r.output_tensor_size,
                    }
                    for r in self.results
                ],
                f,
                indent=2,
            )

        # Generate plots
        self.plot_results(output_dir)

        # Generate summary
        self.generate_summary(output_dir)

    def plot_results(self, output_dir: str):
        """Generate visualization plots of profiling results."""
        if not self.results:
            print("No results to plot")
            return

        # Extract data for plotting
        batch_sizes = [r.input_size[0] for r in self.results]
        max_atoms = [r.input_size[1] for r in self.results]
        forward_times = [r.forward_time for r in self.results]
        throughputs = [r.throughput for r in self.results]
        memory_usage = [r.memory_usage for r in self.results]

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Throughput vs batch size
        axes[0, 0].scatter(batch_sizes, throughputs, alpha=0.7)
        axes[0, 0].set_xlabel("Batch Size")
        axes[0, 0].set_ylabel("Throughput (molecules/s)")
        axes[0, 0].set_title("Throughput vs Batch Size")
        axes[0, 0].grid(True)

        # Forward time vs batch size
        axes[0, 1].scatter(batch_sizes, forward_times, alpha=0.7)
        axes[0, 1].set_xlabel("Batch Size")
        axes[0, 1].set_ylabel("Forward Time (s)")
        axes[0, 1].set_title("Forward Time vs Batch Size")
        axes[0, 1].grid(True)

        # Memory usage vs batch size
        axes[1, 0].scatter(batch_sizes, memory_usage, alpha=0.7)
        axes[1, 0].set_xlabel("Batch Size")
        axes[1, 0].set_ylabel("Memory Usage (MB)")
        axes[1, 0].set_title("Memory Usage vs Batch Size")
        axes[1, 0].grid(True)

        # Throughput vs max atoms
        axes[1, 1].scatter(max_atoms, throughputs, alpha=0.7)
        axes[1, 1].set_xlabel("Max Atoms")
        axes[1, 1].set_ylabel("Throughput (molecules/s)")
        axes[1, 1].set_title("Throughput vs Max Atoms")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "profile_plots.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_summary(self, output_dir: str):
        """Generate a summary report."""
        if not self.results:
            print("No results to summarize")
            return

        # Calculate statistics
        forward_times = [r.forward_time for r in self.results]
        throughputs = [r.throughput for r in self.results]
        memory_usage = [r.memory_usage for r in self.results]

        summary = {
            "total_runs": len(self.results),
            "device": self.device,
            "statistics": {
                "forward_time": {
                    "mean": np.mean(forward_times),
                    "std": np.std(forward_times),
                    "min": np.min(forward_times),
                    "max": np.max(forward_times),
                },
                "throughput": {
                    "mean": np.mean(throughputs),
                    "std": np.std(throughputs),
                    "min": np.min(throughputs),
                    "max": np.max(throughputs),
                },
                "memory_usage": {
                    "mean": np.mean(memory_usage),
                    "std": np.std(memory_usage),
                    "min": np.min(memory_usage),
                    "max": np.max(memory_usage),
                },
            },
            "best_performance": {
                "fastest_forward": min(self.results, key=lambda x: x.forward_time),
                "highest_throughput": max(self.results, key=lambda x: x.throughput),
                "lowest_memory": min(self.results, key=lambda x: x.memory_usage),
            },
        }

        # Save summary
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        print("\n" + "=" * 50)
        print("E3GNN PROFILING SUMMARY")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Total runs: {summary['total_runs']}")
        print(
            f"Average forward time: {summary['statistics']['forward_time']['mean']:.4f}s ± {summary['statistics']['forward_time']['std']:.4f}s"
        )
        print(
            f"Average throughput: {summary['statistics']['throughput']['mean']:.1f} mol/s ± {summary['statistics']['throughput']['std']:.1f} mol/s"
        )
        print(
            f"Average memory usage: {summary['statistics']['memory_usage']['mean']:.1f} MB ± {summary['statistics']['memory_usage']['std']:.1f} MB"
        )

        best = summary["best_performance"]
        print(f"\nBest performance:")
        print(
            f"  Fastest forward: {best['fastest_forward'].forward_time:.4f}s (batch_size={best['fastest_forward'].input_size[0]}, max_atoms={best['fastest_forward'].input_size[1]})"
        )
        print(
            f"  Highest throughput: {best['highest_throughput'].throughput:.1f} mol/s (batch_size={best['highest_throughput'].input_size[0]}, max_atoms={best['highest_throughput'].input_size[1]})"
        )
        print(
            f"  Lowest memory: {best['lowest_memory'].memory_usage:.1f} MB (batch_size={best['lowest_memory'].input_size[0]}, max_atoms={best['lowest_memory'].input_size[1]})"
        )


def main():
    """Main profiling function."""
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize profiler
    profiler = E3GNNProfiler(device=device)

    # Define configurations to test
    configs = [
        {
            "hidden_nf": 64,
            "n_layers": 3,
            "message_cutoff": 5,
            "torch_emb": False,
            "residual": False,
            "dropout": 0.0,
        },
        {
            "hidden_nf": 128,
            "n_layers": 4,
            "message_cutoff": 5,
            "torch_emb": False,
            "residual": False,
            "dropout": 0.0,
        },
        {
            "hidden_nf": 256,
            "n_layers": 5,
            "message_cutoff": 8,
            "torch_emb": False,
            "residual": True,
            "dropout": 0.1,
        },
        {
            "hidden_nf": 128,
            "n_layers": 4,
            "message_cutoff": 5,
            "torch_emb": True,  # Test torch embedding
            "residual": False,
            "dropout": 0.0,
        },
    ]

    # Define input sizes to test
    batch_sizes = [1, 4, 8, 16, 32, 64]
    max_atoms_list = [10, 20, 30, 50]

    print("Starting E3GNN profiling...")

    # Profile different configurations
    profiler.profile_configurations(configs, batch_sizes, max_atoms_list)

    # Profile scaling behavior
    base_config = {
        "hidden_nf": 128,
        "n_layers": 4,
        "message_cutoff": 5,
        "torch_emb": False,
        "residual": False,
        "dropout": 0.0,
    }
    profiler.profile_scaling_behavior(base_config, max_batch_size=128, max_atoms=50)

    # Profile memory efficiency
    memory_result = profiler.profile_memory_efficiency(
        base_config, batch_size=32, max_atoms=50
    )
    print(f"\nMemory efficiency result:")
    print(f"  Input tensor size: {memory_result.input_tensor_size:.2f} MB")
    print(f"  Output tensor size: {memory_result.output_tensor_size:.2f} MB")
    print(f"  GPU memory allocated: {memory_result.gpu_memory_allocated:.2f} MB")
    print(f"  GPU memory reserved: {memory_result.gpu_memory_reserved:.2f} MB")

    # Generate report
    profiler.generate_report()

    print(
        "\nProfiling completed! Check 'e3gnn_profile_results' directory for detailed results."
    )


if __name__ == "__main__":
    main()
