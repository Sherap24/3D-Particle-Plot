# 3D & 2D Particle Plot

A Python-based visualization tool for 3D and 2D scatter plots from event-specific dataset entries. This project uses `matplotlib` and `pandas` for interactive data exploration.

## Features

- **Data Cleaning**: Automatically processes tab-delimited datasets, handling missing or invalid values.
- **Interactive Plots**: Visualize data with:
  - A 3D scatter plot.
  - A 2D scatter plot of Y vs. X.
  - A 2D scatter plot of Y and X vs. Z.
- **Event Navigation**: Navigate between event-specific plots using `Next` and `Previous` buttons.

## Prerequisites

Ensure you have Python 3.10 or higher installed. You'll also need the following libraries:

- `pandas`
- `matplotlib`

Install the dependencies using pip:

```bash
pip install pandas matplotlib
```

## Dataset Format

The program expects a tab-delimited dataset (`Data file.txt`) with the following columns:

| Column Name | Description                 |
|-------------|-----------------------------|
| `event`     | Event number (integer)      |
| `x`         | X-coordinate (float)        |
| `y`         | Y-coordinate (float)        |
| `z`         | Z-coordinate (float)        |
| `tb`        | Timestamp (optional)        |
| `q`         | Quality metric (optional)   |

Ensure the file is formatted correctly to avoid errors. **You will need to provide your own dataset. This code currently works for `Data file.txt` located in the same directory as the script.**

## How to Use

1. Clone the repository or download the script directly:

    ```bash
    git clone https://github.com/Sherap24/3D-Particle-Plot.git
    cd 3D-Particle-Plot
    ```

2. Place your dataset (`Data file.txt`) in the same directory as the script.

3. Run the script:

    ```bash
    python Updated.py
    ```

4. Use the `Next` and `Previous` buttons to navigate through the events.

## Project Structure

```
3D-Particle-Plot/
│
├── Updated.py         # Main Python script for plotting
├── Data file.txt      # Example dataset (optional, if sharing an example)
└── README.md          # This README file
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push to your branch.
4. Open a pull request describing your changes.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code.
