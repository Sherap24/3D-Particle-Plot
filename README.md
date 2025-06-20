# Particle Physics Event Analysis & Classification

A comprehensive Python-based toolkit for visualizing, analyzing, and classifying particle physics events from the X-17 boson search experiment. This project combines interactive 3D visualization tools with machine learning approaches for event classification, particularly focusing on identifying "seagull" patterns in particle tracks.

## Project Overview

This repository contains tools developed for analyzing particle physics data from an Active Target Time Projection Chamber (AT-TPC) experiment searching for the hypothetical X-17 boson. The project has evolved from simple visualization to sophisticated data processing and machine learning classification.

### Current Challenges & Ongoing Work

⚠️ **Data Quality & Classification Challenges**: Our primary struggle has been achieving reliable classification of particle events, particularly distinguishing "seagull" events from background noise. The raw experimental data contains significant noise and varied event topologies that make classification challenging.

🔄 **Iterative Data Cleaning Process**: The `Newest.py` script represents our latest attempt at intelligent data cleaning, but this remains an active area of development. We expect this file to undergo frequent updates as we refine our understanding of the optimal preprocessing pipeline.

🎯 **Classification Goals**: We aim to build a robust classifier to distinguish between:
- **Seagull events**: Characteristic wing-like patterns indicating potential signal events
- **Background events**: Noise, triplets, and other non-signal patterns

## Features

### Interactive Visualization (`GUI new.py`)
- **Enhanced 3D Visualization**: Interactive 3D scatter plots with clustering analysis
- **DBSCAN Clustering**: Real-time clustering with adjustable parameters
- **Multiple View Modes**: 3D plots, 2D projections, and cluster statistics
- **Advanced Navigation**: Event browsing with keyboard shortcuts and direct event search
- **Noise Toggle**: Option to show/hide noise points for cleaner visualization
- **Parameter Tuning**: Interactive controls for clustering parameters (eps, min_samples)

### Data Preprocessing (`Newest.py`)
- **Seagull-Focused Filtering**: Specialized algorithms to preserve seagull-like patterns
- **Adaptive Z-Range Filtering**: Dynamic filtering based on main cluster characteristics
- **Noise Reduction**: Intelligent background noise removal while preserving signal
- **Main Cluster Detection**: Automated identification of primary event features
- **Quality Metrics**: Comprehensive statistics on data cleaning effectiveness

### Legacy Features
- **Event Navigation**: Navigate between events using buttons or keyboard
- **Multi-Plot Display**: Simultaneous 3D and 2D visualizations
- **Data Validation**: Automatic handling of missing or invalid data points

## Installation & Prerequisites

Ensure you have Python 3.8 or higher installed. Required dependencies:

```bash
pip install pandas matplotlib numpy scikit-learn
```

For enhanced visualization features:
```bash
pip install seaborn
```

## Dataset Format

The program expects space-delimited data files with the following structure:

### Original Format (`Data file.txt`)
| Column | Description |
|--------|-------------|
| `a` | Event number (integer) |
| `b` | Secondary identifier |
| `c` | Tertiary identifier |
| `x` | X-coordinate (float) |
| `y` | Y-coordinate (float) |
| `z` | Z-coordinate (float) |
| `tb` | Time bucket/timestamp |
| `q` | Charge information |

### Cleaned Format (`Cleaned_Data.txt`)
| Column | Description |
|--------|-------------|
| `a` | Event number (integer) |
| `x` | X-coordinate (float) |
| `y` | Y-coordinate (float) |
| `z` | Z-coordinate (float) |
| `tb` | Time bucket/timestamp |
| `q` | Charge information |

## Usage

### 1. Data Cleaning (Recommended First Step)

```bash
python Newest.py
```

This will:
- Process raw data from `Data file.txt`
- Apply seagull-focused filtering algorithms
- Generate `Cleaned_Data.txt` optimized for classification
- Provide detailed statistics on noise reduction

**Note**: The cleaning algorithm is under active development. Results may vary between versions as we improve our understanding of optimal preprocessing.

### 2. Interactive Visualization

```bash
python "GUI new.py"
```

Features:
- Navigate events with Next/Previous buttons or arrow keys
- Adjust clustering parameters in real-time
- Toggle noise visibility
- View cluster statistics and 2D projections
- Search for specific events

### Keyboard Shortcuts
- `←/→` or `P/N`: Navigate between events
- `Space`: Next event
- Direct event search via text input

## Project Structure

```
particle-physics-analysis/
│
├── GUI new.py           # Enhanced interactive visualization tool
├── Newest.py            # Data cleaning & preprocessing (actively updated)
├── Data file.txt        # Raw experimental data (user-provided)
├── Cleaned_Data.txt     # Processed data (generated by Newest.py)
├── README.md            # This file
│
└── archive/             # Previous versions and experimental code
    ├── Updated.py       # Legacy visualization tool
    ├── Feb 5.py         # Clustering experiments
    ├── Feb 12.py        # PointNet implementation attempts
    └── Model.py         # Ensemble classification models
```

## Current Research Status

### ✅ Completed
- Interactive 3D visualization with clustering
- Basic data preprocessing pipeline
- Event navigation and parameter tuning interfaces

### 🔄 In Progress (High Priority)
- **Data cleaning optimization** (Newest.py updates expected weekly)
- Seagull event detection algorithms
- Classification model training pipeline

### 📋 Planned
- Automated hyperparameter tuning for clustering
- Integration of machine learning classification
- Batch processing for large datasets
- Statistical analysis of event characteristics

## Known Issues & Limitations

1. **Classification Accuracy**: Current models struggle with high false positive rates
2. **Data Quality**: Raw experimental data contains significant noise requiring ongoing preprocessing refinement
3. **Parameter Sensitivity**: Clustering results are sensitive to DBSCAN parameters
4. **Event Diversity**: Wide variety of event topologies makes unified classification challenging

## Contributing

We welcome contributions, especially in the following areas:
- Data preprocessing and noise reduction algorithms
- Classification model improvements
- Visualization enhancements
- Documentation and testing

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement-name`)
3. Test your changes with the provided datasets
4. Update documentation as needed
5. Submit a pull request with detailed description

### Reporting Issues
When reporting issues, please include:
- Python version and dependency versions
- Sample data or event numbers that demonstrate the issue
- Expected vs. actual behavior
- Any error messages or unexpected outputs

## Research Context

This work is part of ongoing research into the hypothetical X-17 boson, investigating anomalies in particle decay patterns. The classification of "seagull" events is crucial for distinguishing potential signal events from background processes in the experimental data.

## Acknowledgments

- Mississippi State University Physics Department
- FRIB (Facility for Rare Isotope Beams) collaboration
- University of Notre Dame Nuclear Science Laboratory

## License

This project is released under the MIT License. See LICENSE file for details.

---

**⚠️ Active Development Notice**: This codebase is under active development. The data cleaning pipeline (`Newest.py`) undergoes frequent updates as we refine our preprocessing algorithms. Check commit history for the latest improvements and known issues.