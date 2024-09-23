# limited data

To reproduce, run the scripts found in the ./scripts directory. 

1. To reproduce the tuning results, use:
   a. ASHA: scripts/run_main_ray.sh
   b. MTPE: scripts/run_optuna.sh

  specify the input arguments accordingly

2. To Extract all models of the Pareto front run scripts/run_main_pareto.sh

3. To reproduce the training runs, use:
   a. Ray models: scripts/run_main_pareto.sh
   b. All others: scripts/run_main.sh
