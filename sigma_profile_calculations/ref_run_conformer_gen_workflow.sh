#!/bin/bash
#SBATCH --job-name="orca_conformer_pipeline"	# job name
#SBATCH --output="my_output_orca_conformer_gen_pipeline_%j" # output file
#SBATCH --error="my_error_orca_conformer_gen_pipeline_%j" # error file
#SBATCH --nodes=1 #number of nodes
#SBATCH --ntasks=32 #since --nodes = 1, --ntasks-per-node = --ntasks
#SBATCH --mem=32G #memory requested
#SBATCH -t 10-00:00 #total time, days-hours:minutes (can also use hours:minutes:seconds or days-hours or days-hours:minutes:seconds)

#REFERENCE SCRIPT for running conformer generation pipeline to create the sigma profile for a given input structure using openCOSMO-RS_conformer_pipeline and ORCA

#### DEFINE VARIABLES ####
#jobscriptname = name of this file
jobscriptname=ref_run_conformer_gen_workflow.sh

mol_name=1-ethyl-3-methylimidazolium #name of structure will use in conformer gen workflow

#input parameters for creating the structure input file for openCOSMO-RS_conformer_pipeline
mol_smiles_str="CC[n+]1ccn(C)c1"
xyz_file=0 #input is 0 if don't want to use an xyz file
mol_charge=1
geom_opt=True #input is True if want to do geometry optimization

#folder where results will be stored
#typically, don't have to change these lines unless saving results in a different directory
results_folder=$SLURM_SUBMIT_DIR/conformer_gen_${mol_name}_current_job
results_folder_final_name=$SLURM_SUBMIT_DIR/conformer_gen_${mol_name} #Name of folder when job is completed

#path to code for generating structure input file for conformer pipeline
#typically, don't have to change this line unless python file is moved
gen_structure_input_code=$SLURM_SUBMIT_DIR/createConformerGenInputFile.py

#path to structure input file created using python code
#typically, don't have to change this line unless modify code for generating the structure input file
structure_input_file=$results_folder/conformer_gen_input_${mol_name}.inp

#### LOAD MODULES ####
#load modules
module purge # requested modules to run the jobs
module load gnu12/12.2.0 #need to load this in order to load openmpi4/4.1.5 (based on information from module spider openmpi4/4.1.5)
module load openmpi4/4.1.5
module load python
source $HOME/il_ionicity_molar_conductivity_project/bin/activate

echo "Start: `date` on `hostname`"
echo "Home directory: $HOME"
echo "Current directory: $PWD"

#### CREATE FOLDER FOR STORING RESULTS & INPUT FILE FOR CONFORMER PIPELINE ####
#First create folder 
#First create folder to store results
mkdir $results_folder

#copy job script to the folder where results will be stored
cp $jobscriptname $results_folder

#Create conformer generation input file & save it in folder where results will be stored
python -u $gen_structure_input_code -path_to_save_input_file=$results_folder -structure_name=$mol_name -smiles_str=$mol_smiles_str -xyz_filename=$xyz_file -charge=$mol_charge -geometry_optimization=$geom_opt

#move to folder results will be stored to
cd $results_folder


#copy data to /tmp (local directory) on compute node
MYTMP=/tmp/$USER/$SLURM_JOB_ID
/usr/bin/mkdir -p $MYTMP || exit $?
echo "Copying data to /tmp on the compute node ($MYTMP)..."
cp -r $results_folder/* $MYTMP || exit $? #copy files or exit w/error code
cd $MYTMP #move to /tmp on compute node

# run calculation/simulation on compute node
#main command:
conformer_gen_folder_path=/$HOME/packages/openCOSMO-RS_conformer_pipeline-main
python $conformer_gen_folder_path/ConformerGenerator.py --structures_file $structure_input_file --cpcm_radii $conformer_gen_folder_path/cpcm_radii.inp --n_cores 2

echo "Copying data back to $results_folder..."
cp -rp $MYTMP/* $results_folder || exit $?
rm -rf $MYTMP #remove data from compute node /tmp space

cd $results_folder #leave /tmp on compute node
mv $results_folder $results_folder_final_name
cd $SLURM_SUBMIT_DIR #move back to home directory
echo "Moved back to $SLURM_SUBMIT_DIR"
mv my_output_orca_conformer_gen_pipeline_$SLURM_JOB_ID $results_folder_final_name
mv my_error_orca_conformer_gen_pipeline_$SLURM_JOB_ID $results_folder_final_name

echo "End: `date` on `hostname`"
deactivate

