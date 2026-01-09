#!/bin/bash
#SBATCH --job-name=ref_load_eval_optimal_models  # job name
#SBATCH --output="my_output_load_eval_optimal_models_%j" # output file
#SBATCH --error="my_error_load_eval_optimal_models_%j" # error file
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=32
#SBATCH --time=20-00:00:00 # total time, hours:minutes:seconds (can also use days-hours:minutes or days-hours or days-hours:minutes:seconds or minutes or minutes:seconds)


##This job runs with 1 node 

echo "Start: `date` on `hostname`"

module purge # requested modules to run the jobs
module load python
source $HOME/il_ionicity_molar_conductivity_project/bin/activate

#Define parameters to pass into code & load models/run code
#Example - use input feature set: S_i descriptors
input_features_to_use='S_i' #could also have sigma_profiles, moment_waps_wans, RDKit, etc.

## Conductivity Prediction ##
ionicRadiiEstimationMethod='NA' #if ionicity, use RDKitDCLV_vdWVolAndSA
molarConductivityOrIonicity='conductivity' #or can use ionicity

#load linear L1 model w/optimal hyperparameter value & then evaluate model
modelType='linear_l1'
optimal_hyperparam_val=0.001
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/optimal_model_example_with_loading_models/${molarConductivityOrIonicity}/${modelType}/

python -u load_eval_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -input_features_to_use $input_features_to_use -opt_hyperparameter_val $optimal_hyperparam_val

#load XGBoost model w/optimal hyperparameter value & then evaluate model
modelType='xgboost'
optimal_max_depth=13
optimal_n_estimators=860
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/optimal_model_example_with_loading_models/${molarConductivityOrIonicity}/${modelType}/

python -u load_eval_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -input_features_to_use $input_features_to_use -opt_hyperparameter_val $optimal_max_depth -opt_hyperparameter_val2 $optimal_n_estimators

## Ionicity Prediction ##
ionicRadiiEstimationMethod='RDKitDCLV_vdWVolAndSA' #if conductivity, use NA
molarConductivityOrIonicity='ionicity' #or can use conductivity

#train linear L1 model w/optimal hyperparameter value & then evaluate model
modelType='linear_l1'
optimal_hyperparam_val=0.0001
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/optimal_model_example_with_loading_models/${molarConductivityOrIonicity}/${modelType}/

python -u load_eval_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -input_features_to_use $input_features_to_use -opt_hyperparameter_val $optimal_hyperparam_val

#train XGBoost model w/optimal hyperparameter value & then evaluate model
modelType='xgboost'
optimal_max_depth=11
optimal_n_estimators=130
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/optimal_model_example_with_loading_models/${molarConductivityOrIonicity}/${modelType}/

python -u load_eval_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -input_features_to_use $input_features_to_use -opt_hyperparameter_val $optimal_max_depth -opt_hyperparameter_val2 $optimal_n_estimators


echo "End: `date` on `hostname`"
deactivate
