# MarkovChain

This repository contains a Markov chain model for simulating dairy herd dynamics, economics, and methane emissions under different management scenarios, including the use of methane-reducing feed additives.

## Features

- Simulates herd evolution using a Markov transition matrix
- Models parity, lactation, pregnancy, culling, and replacement
- Calculates monthly and cumulative net present value (NPV) for the herd
- Estimates methane emissions and the impact of feed additives
- Supports incentive policy analysis for methane abatement
- Visualizes herd parity structure and economic outcomes

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MarkovChain.git
   cd MarkovChain
   ```

2. **Run the main simulation:**
   ```bash
   python main.py
   ```

3. **Outputs:**
   - Figures showing parity group evolution and NPV comparisons are saved to the specified output directory.
   - Console prints summary statistics for NPV and herd structure.

## Customization

- Adjust model parameters (e.g., culling rates, milk yield, additive effects) in `main.py`.
- Change incentive policy logic in the `find_best_incentive` function.
- Modify output paths and plotting options as needed.

## Requirements

- Python 3.x
- numpy
- matplotlib

Install dependencies with:
```bash
pip install numpy matplotlib
```

## License

MIT License

## Contact

For questions or suggestions, please contact hh598@cornell.edu.