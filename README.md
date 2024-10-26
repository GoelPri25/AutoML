# AutoML

![image](https://github.com/user-attachments/assets/4604df7e-f010-4d12-98dc-72ce9e7ed7e7)
From: Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011). Sequential model-based optimization for general algorithm configuration. In Learning and Intelligent Optimization: 5th International Conference, LION 5, Rome, Italy, January 17-21, 2011. Selected Papers 5 (pp. 507-523). Springer Berlin Heidelberg.

## Instructions

To run experiments using AutoML, follow these steps.

### Prerequisites

1. Ensure you have all required dependencies installed. Run the following command to install dependencies:

   ```bash
   pip install -r requirements.txt
2. Surrogate via random search
   ```bash
   python example_run_experiment.py

3. SMBO
   ```bash
   python smbo.py

4. Successive Halving
     ```bash
   python successive_halving.py

5. Compare plot for SMBO vs Random search
     ```bash
   python smbo_vs_randomsearch.py
6. Compare plot for SMBO vs Successive halving
    ```bash
   python smbo_vs_sh.py
