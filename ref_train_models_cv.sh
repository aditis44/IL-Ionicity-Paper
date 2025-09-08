#!/bin/bash
#SBATCH --job-name=ref_train_models_cv    # job name
#SBATCH --output="my_output_train_models_cv_%j" # output file
#SBATCH --error="my_error_train_models_cv_%j" # error file
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=32
#SBATCH --time=20-00:00:00 # total time, hours:minutes:seconds (can also use days-hours:minutes or days-hours or days-hours:minutes:seconds or minutes or minutes:seconds)

##This job runs with 1 node 

echo "Start: `date` on `hostname`"

#load modules
module purge # requested modules to run the jobs
module load python
source $HOME/il_ionicity_molar_conductivity_project/bin/activate

#Define parameters to pass into code & train models/run code
#Example - use input feature set: S_i descriptors
input_features_to_use='S_i' #could also have sigma_profiles, moment_waps_wans, RDKit, etc.

## Conductivity Prediction ##
ionicRadiiEstimationMethod='NA' #if ionicity, use RDKitDCLV_vdWVolAndSA
molarConductivityOrIonicity='conductivity' #or can use ionicity
compareToDummyRegressor='True'
doCrossValidation='True'

#train Linear L1 model w/input S_i features
modelType='linear_l1'
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/cross_validation_example/${molarConductivityOrIonicity}/${modelType}/
python -u train_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod  -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -compare_to_dummy_regressor $compareToDummyRegressor -do_cross_validation $doCrossValidation -input_features_to_use $input_features_to_use

#train XGBoost model w/input S_i features
modelType='xgboost'
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/cross_validation_example/${molarConductivityOrIonicity}/${modelType}/

python -u train_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod  -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -compare_to_dummy_regressor $compareToDummyRegressor -do_cross_validation $doCrossValidation -input_features_to_use $input_features_to_use

## Ionicity Prediction ##
ionicRadiiEstimationMethod='RDKitDCLV_vdWVolAndSA' #if conductivity, use NA
molarConductivityOrIonicity='ionicity' #or can use conductivity
compareToDummyRegressor='True'
doCrossValidation='True'

#train Linear L1 model w/input S_i features
modelType='linear_l1'
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/cross_validation_example/${molarConductivityOrIonicity}/${modelType}/
python -u train_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod  -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -compare_to_dummy_regressor $compareToDummyRegressor -do_cross_validation $doCrossValidation -input_features_to_use $input_features_to_use

#train XGBoost model w/input S_i features
modelType='xgboost'
directoryToSaveResultsTo=$SLURM_SUBMIT_DIR/cross_validation_example/${molarConductivityOrIonicity}/${modelType}/

python -u train_models.py -molar_conductivity_or_ionicity $molarConductivityOrIonicity -ionic_radii_estimation_method $ionicRadiiEstimationMethod  -directory_to_save_results $directoryToSaveResultsTo -model_type $modelType -compare_to_dummy_regressor $compareToDummyRegressor -do_cross_validation $doCrossValidation -input_features_to_use $input_features_to_use


echo "End: `date` on `hostname`"
deactivate

