#!/usr/bin/env python3
"""
Synthetic data generator for EPI project.
Creates realistic synthetic patient data that maintains the same structure
as the original dataset while protecting privacy.
"""

import h5py
import numpy as np
import os
from pathlib import Path

def generate_synthetic_ehr_data(
    n_patients=5000,
    n_diagnoses=500,
    n_features=4,
    output_dir="synthetic_data",
    random_seed=42
):
    """
    Generate synthetic EHR data with the same structure as original EPI dataset.

    Parameters:
    -----------
    n_patients : int
        Number of synthetic patients
    n_diagnoses : int
        Number of diagnosis codes (comorbidities)
    n_features : int
        Number of features per node (default: 4 for age, gender, race, other)
    output_dir : str
        Directory to save synthetic data files
    random_seed : int
        Random seed for reproducibility
    """

    np.random.seed(random_seed)

    print(f"🔬 Generating synthetic EHR data...")
    print(f"   📊 Patients: {n_patients:,}")
    print(f"   🏥 Diagnoses: {n_diagnoses:,}")
    print(f"   📋 Features per node: {n_features}")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate node features (x)
    total_nodes = n_diagnoses + n_patients
    print(f"   🔢 Total nodes: {total_nodes:,}")

    # Features for diagnosis nodes (first n_diagnoses rows)
    # These are typically just indices or simple categorical features
    diag_features = np.zeros((n_diagnoses, n_features), dtype=np.float32)
    diag_features[:, 0] = np.arange(n_diagnoses)  # Diagnosis ID

    # Features for patient nodes (remaining rows)
    # Generate realistic demographic and clinical features
    patient_features = np.zeros((n_patients, n_features), dtype=np.float32)

    # Feature 0: Age (18-100 years)
    patient_features[:, 0] = np.random.normal(55, 18, n_patients)
    patient_features[:, 0] = np.clip(patient_features[:, 0], 18, 100)

    # Feature 1: Gender (0=Unknown, 1=Male, 2=Female)
    patient_features[:, 1] = np.random.choice([1, 2], n_patients, p=[0.48, 0.52])

    # Feature 2: Race/Ethnicity (1=White, 2=Black, 3=Hispanic, 4=Asian, 5=Native, 6=Other)
    patient_features[:, 2] = np.random.choice(
        [1, 2, 3, 4, 5, 6],
        n_patients,
        p=[0.6, 0.15, 0.12, 0.08, 0.02, 0.03]
    )

    # Feature 3: Additional clinical indicator (normalized)
    patient_features[:, 3] = np.random.exponential(2.0, n_patients)
    patient_features[:, 3] = np.clip(patient_features[:, 3], 0, 10)

    # Combine all features
    x = np.vstack([diag_features, patient_features]).astype(np.float32)
    print(f"   ✅ Generated node features: {x.shape}")

    # Generate labels (y) - binary epilepsy prediction
    # Realistic class imbalance (~20% positive cases)
    y = np.random.choice([0, 1], n_patients, p=[0.78, 0.22]).astype(np.int64)
    print(f"   📊 Label distribution: {np.sum(y)} positive ({100*np.mean(y):.1f}%), {len(y)-np.sum(y)} negative")

    # Generate realistic edges (patient-diagnosis relationships)
    print(f"   🔗 Generating patient-diagnosis edges...")

    # Each patient has connections to multiple diagnoses
    # More common diagnoses appear more frequently

    # Create diagnosis popularity (some diagnoses are much more common)
    diag_popularity = np.random.zipf(1.5, n_diagnoses)
    diag_popularity = diag_popularity / np.sum(diag_popularity)

    edges = []
    edge_weights = []

    for patient_idx in range(n_patients):
        # Number of diagnoses per patient (1-20, most have 3-8)
        n_patient_diagnoses = min(20, max(1, int(np.random.gamma(2, 2))))

        # Select diagnoses based on popularity and some randomness
        selected_diagnoses = np.random.choice(
            n_diagnoses,
            size=n_patient_diagnoses,
            replace=False,
            p=diag_popularity
        )

        # Create edges: patient -> diagnosis
        patient_node_idx = n_diagnoses + patient_idx  # Offset by diagnosis nodes

        for diag_idx in selected_diagnoses:
            # Edge from patient to diagnosis
            edges.append([patient_node_idx, diag_idx])

            # Edge weight represents strength of association (e.g., time since diagnosis)
            edge_weight = np.random.exponential(2.0)  # Similar distribution to original
            edge_weights.append(edge_weight)

    # Convert to tensors
    edge_index = np.array(edges).T.astype(np.int64)  # Shape: (2, n_edges)
    edge_attr = np.array(edge_weights).reshape(-1, 1).astype(np.float32)  # Shape: (n_edges, 1)

    print(f"   🔗 Generated {edge_index.shape[1]:,} edges")
    print(f"   📈 Average edges per patient: {edge_index.shape[1]/n_patients:.1f}")

    # Save different file versions (matching original naming convention)
    file_configs = [
        ("20", 20),
        ("50", 50),
        ("100", 100),
        ("200", 200),
        ("500", 500)
    ]

    for suffix, diag_count in file_configs:
        if diag_count <= n_diagnoses:
            filename = f"{output_dir}/EPI_heterodata_diag{suffix}_synthetic.h5"

            # Subset data for this configuration
            subset_x = np.vstack([
                x[:diag_count],  # First diag_count diagnosis nodes
                x[n_diagnoses:]  # All patient nodes
            ])

            # Adjust edge indices for subset
            mask = edge_index[1] < diag_count  # Only keep edges to included diagnoses
            subset_edge_index = edge_index[:, mask].copy()
            subset_edge_attr = edge_attr[mask].copy()

            # Adjust patient node indices in edges
            subset_edge_index[0] = subset_edge_index[0] - (n_diagnoses - diag_count)

            print(f"   💾 Saving {filename}")
            print(f"      📊 Nodes: {subset_x.shape[0]:,}, Edges: {subset_edge_index.shape[1]:,}")

            with h5py.File(filename, 'w') as f:
                f.create_dataset('x', data=subset_x)
                f.create_dataset('edge_index', data=subset_edge_index)
                f.create_dataset('edge_attr', data=subset_edge_attr)
                f.create_dataset('y', data=y)

    # Also create the main file
    main_filename = f"{output_dir}/EPI_heterodata_synthetic.h5"
    print(f"   💾 Saving main file: {main_filename}")

    with h5py.File(main_filename, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('edge_index', data=edge_index)
        f.create_dataset('edge_attr', data=edge_attr)
        f.create_dataset('y', data=y)

    print(f"✅ Synthetic data generation complete!")
    print(f"   📁 Files saved in: {output_dir}/")

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'n_patients': n_patients,
        'n_diagnoses': n_diagnoses
    }

def validate_synthetic_data(data_dir="synthetic_data"):
    """Validate that synthetic data has correct structure and properties."""

    print(f"🔍 Validating synthetic data in {data_dir}/...")

    # Check main file
    main_file = f"{data_dir}/EPI_heterodata_synthetic.h5"

    if not os.path.exists(main_file):
        print(f"❌ Main file not found: {main_file}")
        return False

    with h5py.File(main_file, 'r') as f:
        x = f['x'][:]
        edge_index = f['edge_index'][:]
        edge_attr = f['edge_attr'][:]
        y = f['y'][:]

        print(f"   📊 Node features (x): {x.shape}")
        print(f"   🔗 Edge index: {edge_index.shape}")
        print(f"   ⚖️ Edge attributes: {edge_attr.shape}")
        print(f"   🎯 Labels (y): {y.shape}")

        # Validation checks
        checks_passed = 0
        total_checks = 6

        # Check 1: Edge indices are valid
        if edge_index.max() < x.shape[0] and edge_index.min() >= 0:
            print(f"   ✅ Edge indices are valid (0 to {x.shape[0]-1})")
            checks_passed += 1
        else:
            print(f"   ❌ Invalid edge indices: min={edge_index.min()}, max={edge_index.max()}")

        # Check 2: Labels are binary
        if set(y) <= {0, 1}:
            print(f"   ✅ Labels are binary (0/1)")
            checks_passed += 1
        else:
            print(f"   ❌ Labels not binary: unique values = {set(y)}")

        # Check 3: Edge attributes are positive
        if edge_attr.min() >= 0:
            print(f"   ✅ Edge attributes are non-negative")
            checks_passed += 1
        else:
            print(f"   ❌ Negative edge attributes found: min = {edge_attr.min()}")

        # Check 4: Reasonable class balance
        pos_rate = np.mean(y)
        if 0.1 <= pos_rate <= 0.4:
            print(f"   ✅ Reasonable class balance: {pos_rate:.1%} positive")
            checks_passed += 1
        else:
            print(f"   ⚠️ Unusual class balance: {pos_rate:.1%} positive")

        # Check 5: Edge count makes sense
        edges_per_patient = edge_index.shape[1] / len(y)
        if 2 <= edges_per_patient <= 50:
            print(f"   ✅ Reasonable edge density: {edges_per_patient:.1f} edges/patient")
            checks_passed += 1
        else:
            print(f"   ⚠️ Unusual edge density: {edges_per_patient:.1f} edges/patient")

        # Check 6: Data types are correct
        dtypes_ok = (x.dtype == np.float32 and
                    edge_index.dtype == np.int64 and
                    edge_attr.dtype == np.float32 and
                    y.dtype == np.int64)
        if dtypes_ok:
            print(f"   ✅ Data types are correct")
            checks_passed += 1
        else:
            print(f"   ❌ Incorrect data types")

        print(f"   📊 Validation score: {checks_passed}/{total_checks} checks passed")

        return checks_passed == total_checks

if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_ehr_data()

    # Validate the generated data
    is_valid = validate_synthetic_data()

    if is_valid:
        print("🎉 Synthetic data generation successful!")
    else:
        print("⚠️ Synthetic data validation found issues.")