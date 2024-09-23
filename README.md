# limited data

To reproduce, load the data and run the scripts found in the ./scripts directory. 

### 1. Load the data:
  follow the instructions provided by https://github.com/tmehari/ssm_ecg.git

### 2. To reproduce the tuning results, use: 
   a. ASHA: scripts/run_main_ray.sh
   b. MTPE: scripts/run_optuna.sh

  specify the input arguments accordingly

### 3. To Extract all models of the Pareto front run:
   a. run scripts/run_main_pareto.sh

### 4. To reproduce the training runs, use:
   a. Ray models: scripts/run_main_pareto.sh
   b. All others: scripts/run_main.sh
