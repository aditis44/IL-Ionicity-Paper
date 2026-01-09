#Example Python code that loads a previously trained ML model and uses it predict ionicity or molar conductivity
#Uses preprocessed dataset created using data from NIST ILThermo database (https://ilthermo.boulder.nist.gov, Kazakov et al., Dong et al. (2007)) & ILThermoPy package (https://github.com/IvanChernyshov/ILThermoPy)
#Note: code currently assumes that saved models are in a folder called "saved_models" (need to edit saved_model_path definition to change path to saved models)

#When predicting conductivity, a log transformation of y values was applied (found to improve model performance) (no log transformation when predicting ionicity)
#Before using models, downselect features by removing features that are highly correlated with each other (|correlation coef| > 0.9)
## Input to models: features that are not highly correlated with each other 

#Imports
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import os
import argparse
import pickle

import createPlotsML
from createPlotsML import *

#Try setting plotting parameters style w/seaborn
sns.set_context("notebook") #also have paper, poster & talk formats
sns.set_style("dark",
			 {
				 "xtick_bottom": True,
				 "ytick_left": True,
				 "xtick.color": "black",
				 "ytick.color": "black",
				 "axes.edgecolor": "black",
				 "figure.dpi": 300,
			 })
#also set default figure size
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams["savefig.bbox"] = 'tight'

# Set a random seed to ensure reproducibility across runs
RNG_SEED = 42
np.random.seed(seed=RNG_SEED)

PATH = os.getcwd()
curr_directory = PATH

#SETUP/INITIALIZE VARIABLES related to what dataset to use (molar conductivity or ionicity - if ionicity, which ionic radii estimation method), what directory to save results to, what model type (linear w/L1 regularization or xgboost) to use
# read in ionic radii estimation method to use (i.e. ionicity dataset to use) & directory to save results to
parser = argparse.ArgumentParser(description="information on which dataset to evaluate models on, what path to save results to, and what model type was used to train")
parser.add_argument('-molar_conductivity_or_ionicity', type=str, dest='molar_conductivity_or_ionicity', help='value to predict (y data to train on) is either conductivity or ionicity')
parser.add_argument('-ionic_radii_estimation_method', type=str, dest='ionic_radii_estimation_method', help='Ionic radii estimation method, options: NA, RDKitDCLV_vdWVolAndSA')
parser.add_argument('-input_features_to_use', type=str, dest='input_features_to_use',help='Set of input features to use (x values) for training the ML model, options: RDKit, RDKit_and_sigma_profiles, sigma_profiles, S_i, RDKit_and_S_i, moment_waps_wans, RDKit_and_moment_waps_wans_desc, S_i_and_moment_waps_wans')
parser.add_argument('-directory_to_save_results', type=str, dest='directory_to_save_results', help='directory where results (csv & plots) will be saved to')
parser.add_argument('-model_type', type=str, dest='model_type', help='type of model to train (options: linear_l1, xgboost)')
parser.add_argument('-opt_hyperparameter_val', type=float, dest='opt_hyperparameter_val', help='optimal hyperparameter (alpha or max_depth) value to use when training model on full training set, will provide results on train & test sets')
parser.add_argument('-opt_hyperparameter_val2', type=int, dest='opt_hyperparameter_val2', help='second optimal hyperparameter (n_estimators) value to use when training XGBoost model on full training set, will provide results on train & test sets')


args = parser.parse_args()
molar_conductivity_or_ionicity = args.molar_conductivity_or_ionicity #whether to predict molar conductivity or predict ionicity
ionic_radii_estimation_method = args.ionic_radii_estimation_method
input_features_to_use = args.input_features_to_use
directory_to_save_results = args.directory_to_save_results
model_type = args.model_type
if(model_type == 'xgboost'): #have 2 hyperparameters that were varied
	opt_max_depth_val = int(args.opt_hyperparameter_val)
	opt_n_estimators_val = int(args.opt_hyperparameter_val2)
	opt_hyperparameter_val = [opt_max_depth_val,opt_n_estimators_val]
else:
	opt_hyperparameter_val = args.opt_hyperparameter_val

	

#### Read in dataset splits, double check there are no overlaps ####
#read_in_dataset_splits reads in the dataset splits created using sklearn GroupShuffleSplit saved to csv files
#returns dictionary with pandas dataframes corresponding to x,y, and dataset_info train, validation split and test split (keys are x_train, x_val, x_test, y_train, y_val, y_test, dataset_info_train, dataset_info_val, dataset_info_test)
def read_in_dataset_splits():
	dataset_folder = 'ILThermo_Preprocessed_Data'
	dataset_type = '_onlyIonicitiesBelow10'		
	
	if(molar_conductivity_or_ionicity != 'conductivity' and molar_conductivity_or_ionicity != 'ionicity'):
		sys.exit(f'Invalid information on target property to predict was provided (input was {molar_conductivity_or_ionicity}, options are conductivity or ionicity), need to edit & rerun')

	if(input_features_to_use == 'RDKit'):
		feature_str = ''
	elif(input_features_to_use == 'RDKit_and_sigma_profiles'):
		feature_str = 'RDKit_and_sigma_profile_desc_'
	elif(input_features_to_use == 'RDKit_and_S_i'):
		feature_str = 'RDKit_and_S_i_desc_'
	elif(input_features_to_use == 'sigma_profiles'):
		feature_str = 'sigma_profile_desc_'
	elif(input_features_to_use == 'S_i'):
		feature_str = 'S_i_desc_'
	elif(input_features_to_use == 'moment_waps_wans'):
		feature_str = 'moment_waps_wans_desc_'
	elif(input_features_to_use == 'S_i_and_moment_waps_wans'):
		feature_str = 'S_i_and_moment_waps_wans_desc_'
	elif(input_features_to_use == 'RDKit_and_moment_waps_wans'):
		feature_str = 'RDKit_and_moment_waps_wans_desc_'
	else:
		sys.exit(f'Invalid information on what input feature set (x) to use (input was {input_features_to_use}, options are RDKit, RDKit_and_sigma_profiles, sigma_profiles, S_i, RDKit_and_S_i, moment_waps_wans, RDKit_and_moment_waps_wans_desc, S_i_and_moment_waps_wans), need to edit & rerun')

	#dataset_info files contain the compound name, SMILES string (for the IL & cation and anion separately), cation family, longest alkyl chain, error in conductivity, viscosity, density measurement
	dataset_info_train_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/dataset_info_train_GroupShuffleSplit{dataset_type}.csv')
	dataset_info_val_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/dataset_info_validation_GroupShuffleSplit{dataset_type}.csv')
	dataset_info_test_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/dataset_info_test_GroupShuffleSplit{dataset_type}.csv')

	dataset_info_train_df = pd.read_csv(dataset_info_train_path_GroupShuffleSplit)
	dataset_info_val_df = pd.read_csv(dataset_info_val_path_GroupShuffleSplit)
	dataset_info_test_df = pd.read_csv(dataset_info_test_path_GroupShuffleSplit)

	# Also read in train/validation/test sets DataFrames w/features & ionicities
	#x_data consists of the RDKit descriptors (for the cation & anion) along with the pressure and temperature for each entry in the dataset
	train_x_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/x_unscaled_train_{feature_str}GroupShuffleSplit{dataset_type}.csv')
	val_x_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/x_unscaled_validation_{feature_str}GroupShuffleSplit{dataset_type}.csv')
	test_x_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/x_unscaled_test_{feature_str}GroupShuffleSplit{dataset_type}.csv')

	#y_data consists of the ionicity for each entry in the dataset
	if(ionic_radii_estimation_method =='RDKitDCLV_vdWVolAndSA'):
		print('Reading in y data/ionicity values found using RDKit DCLV van der Waals volume & surface area for ionic radii estimation: ')
		train_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/y_{molar_conductivity_or_ionicity}_vdw_vol_and_sa_calcAfterConvertCondToScm2PerMolUnits_unscaled_train_GroupShuffleSplit{dataset_type}.csv')
		val_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/y_{molar_conductivity_or_ionicity}_vdw_vol_and_sa_calcAfterConvertCondToScm2PerMolUnits_unscaled_validation_GroupShuffleSplit{dataset_type}.csv')
		test_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits//y_{molar_conductivity_or_ionicity}_vdw_vol_and_sa_calcAfterConvertCondToScm2PerMolUnits_unscaled_test_GroupShuffleSplit{dataset_type}.csv')
	elif(ionic_radii_estimation_method =='NA'):
		if(molar_conductivity_or_ionicity == 'conductivity'):
			print('Reading in y data/conductivity values: ')
			train_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/y_{molar_conductivity_or_ionicity}_unscaled_Scm2PerMolUnits_train_GroupShuffleSplit{dataset_type}.csv')
			val_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits/y_{molar_conductivity_or_ionicity}_unscaled_Scm2PerMolUnits_validation_GroupShuffleSplit{dataset_type}.csv')
			test_y_path_GroupShuffleSplit = os.path.join(PATH, f'{dataset_folder}/ILThermo_Preprocessed_Data_Splits//y_{molar_conductivity_or_ionicity}_unscaled_Scm2PerMolUnits_test_GroupShuffleSplit{dataset_type}.csv')
		else:
			sys.exit(f'Invalid input - no ionic radii estimation method provided even though predicting ionicity/target property is {molar_conductivity_or_ionicity} (available options: RDKitDCLV_vdWVolAndSA for ionicity, NA for conductivity), need to edit & rerun')
	else:
		sys.exit(f'Invalid ionic radii estimation method provided (input: {ionic_radii_estimation_method}, available options: NA (if predicting conductivity), RDKitDCLV_vdWVolAndSA), need to edit & rerun')

	x_data_train_df = pd.read_csv(train_x_path_GroupShuffleSplit)
	x_data_val_df = pd.read_csv(val_x_path_GroupShuffleSplit)
	x_data_test_df = pd.read_csv(test_x_path_GroupShuffleSplit)

	y_data_train_df = pd.read_csv(train_y_path_GroupShuffleSplit)
	y_data_val_df = pd.read_csv(val_y_path_GroupShuffleSplit)
	y_data_test_df = pd.read_csv(test_y_path_GroupShuffleSplit)

	dataset_splits_dict = {}
	dataset_splits_dict['x_train'] = x_data_train_df
	dataset_splits_dict['x_val'] = x_data_val_df
	dataset_splits_dict['x_test'] = x_data_test_df
	dataset_splits_dict['y_train'] = y_data_train_df
	dataset_splits_dict['y_val'] = y_data_val_df
	dataset_splits_dict['y_test'] = y_data_test_df
	dataset_splits_dict['dataset_info_train'] = dataset_info_train_df
	dataset_splits_dict['dataset_info_val'] = dataset_info_val_df
	dataset_splits_dict['dataset_info_test'] = dataset_info_test_df

	return dataset_splits_dict

#verify_no_overlaps_between_splits checks that there are no overlaps between train, test and validation sets
def verify_no_overlaps_between_splits(dataset_splits_dict):
	#Check what the % of entire dataset in training, testing & validation data splits are
	x_data_train_df = dataset_splits_dict['x_train']
	x_data_val_df = dataset_splits_dict['x_val']
	x_data_test_df = dataset_splits_dict['x_test']
	y_data_train_df = dataset_splits_dict['y_train']
	y_data_val_df = dataset_splits_dict['y_val']
	y_data_test_df = dataset_splits_dict['y_test']
	dataset_info_train = dataset_splits_dict['dataset_info_train']
	dataset_info_val = dataset_splits_dict['dataset_info_val']
	dataset_info_test = dataset_splits_dict['dataset_info_test']

	num_entries_train_set = len(dataset_info_train['Compound'])
	num_entries_val_set = len(dataset_info_val['Compound'])
	num_entries_test_set = len(dataset_info_test['Compound'])
	num_entries_in_dataset = num_entries_train_set + num_entries_val_set + num_entries_test_set

	print(f'Total number of entries in dataset: {num_entries_in_dataset} \n')

	print(f'train dataset shape: {dataset_info_train.shape}')
	print(f'% of data in training set: {num_entries_train_set/num_entries_in_dataset}')
	#print('***train_data.head()***')
	#print(dataset_info_train.head(), '\n')

	print(f'validation dataset shape: {dataset_info_val.shape}')
	print(f'% of data in validation set: {num_entries_val_set/num_entries_in_dataset}')
	#print('***validation_data.head()***')
	#print(dataset_info_val.head(), '\n')

	print(f'test dataset shape: {dataset_info_test.shape}\n')
	print(f'% of data in test set: {num_entries_test_set/num_entries_in_dataset}')
	#print('***test_data.head()***') 
	#print(dataset_info_test.head(), '\n')

	#Check train/validation/test sets do not overlap
	train_cmpds_unique = dataset_info_train['Compound'].unique()
	val_cmpds_unique = dataset_info_val['Compound'].unique()
	test_cmpds_unique = dataset_info_test['Compound'].unique()
	num_unique_cmpds = len(train_cmpds_unique) + len(val_cmpds_unique) + len(test_cmpds_unique)

	print(f'Total number of unique compounds: {num_unique_cmpds}')
	print(f'Number of unique training compounds: {len(train_cmpds_unique)}; Estimated # of unique training compounds for {0.7} split: {0.7*num_unique_cmpds:.3f}')
	print(f'Number of unique validation compounds: {len(val_cmpds_unique)}; Estimated # of unique validation compounds for {0.2} split: {0.2*num_unique_cmpds:.3f}')
	print(f'Number of unique testing compounds: {len(test_cmpds_unique)}; Estimated # of unique testing compounds for {0.1} split: {0.1*num_unique_cmpds:.3f}' )

	#Ensure that the lists of compound names do not overlap by checking whether intersection of the 2 sets is not empty
	empty_set = set([])
	print(f'Do training & validation sets overlap?: {set(train_cmpds_unique).intersection(set(val_cmpds_unique)) != empty_set}')
	print(f'Do training & testing sets overlap?: {set(train_cmpds_unique).intersection(set(test_cmpds_unique)) != empty_set}')
	print(f'Do validation & testing sets overlap?: {set(val_cmpds_unique).intersection(set(test_cmpds_unique)) != empty_set}')


	#Check that training, testing & validation data splits do not overlap (duplicate test)
	train_cmpds = set(dataset_info_train['Compound'].unique())
	val_cmpds = set(dataset_info_val['Compound'].unique())
	test_cmpds = set(dataset_info_test['Compound'].unique())

	common_cmpds1 = train_cmpds.intersection(test_cmpds)
	common_cmpds2 = train_cmpds.intersection(val_cmpds)
	common_cmpds3 = test_cmpds.intersection(val_cmpds)

	print(f'# of common compounds in intersection train & test: {len(common_cmpds1)}; common compounds: {common_cmpds1}')
	print(f'# of common compounds in intersection train & validation: {len(common_cmpds2)}; common compounds: {common_cmpds2}')
	print(f'# of common compounds in intersection test & validation: {len(common_cmpds3)}; common compounds: {common_cmpds3}')
	#'''

	#Check that training, testing & validation x/features dataframes do not overlap using pandas
	x_train_val_intersection = pd.merge(x_data_train_df, x_data_val_df, how='inner')
	x_train_test_intersection = pd.merge(x_data_train_df, x_data_test_df, how = 'inner')
	x_val_test_intersection = pd.merge(x_data_val_df, x_data_test_df, how = 'inner')

	print('x_data/features: \n')
	print(f'intersection train & validation: {x_train_val_intersection}\n')
	print(f'intersection train & test: {x_train_test_intersection}\n')
	print(f'intersection validation & test: {x_val_test_intersection}\n')

	print('Shape of x,y train/validation/test sets:')
	print(f'x_train: {x_data_train_df.shape}')
	print(f'x_val: {x_data_val_df.shape}')
	print(f'x_test: {x_data_test_df.shape}')
	print(f'y_train: {y_data_train_df.shape}')
	print(f'y_val: {y_data_val_df.shape}')
	print(f'y_test: {y_data_test_df.shape}')
	#https://www.geeksforgeeks.org/intersection-of-two-dataframe-in-pandas-python/

#get_descriptors_to_keep takes in correlation matrix (pandas dataframe) & print_level (1 = high/additional information printed out as code runs, 0 = low)
#returns list of descriptors to keep based on correlation matrix (for any descriptors i,j w/|correlation coef.| >= 0.9 only descriptor i will be kept)
def get_descriptors_to_keep(correlation_matrix, print_level = 0):
	#identify descriptors w/|correlation| >= 0.9
	descriptor_names = correlation_matrix.columns
	num_descriptors = len(descriptor_names)
	descriptors_to_include = []
	descriptors_to_omit = []
	for i in range(num_descriptors): #for each descriptor i/column i in correlation matrix
		if(descriptor_names[i] not in descriptors_to_omit): #if current descriptor was not found to be highly correlated w/a previous descriptor, then add it
			descriptors_to_include.append(descriptor_names[i]) #include descriptor i
		for j in range(i + 1, num_descriptors): #for j > i
			curr_correlation_coef = correlation_matrix.iloc[i,j]
			if(np.abs(curr_correlation_coef) >= 0.9): #if correlation coefficient >=0.9 or <= -0.9 
				#have high correlation with descriptor i & descriptor j
				#do not include the descriptor associated w/j in list of descriptors w/low correlation (i.e. since descriptors i & j have high correlation)
				descriptors_to_omit.append(descriptor_names[j])
				if(print_level == 1):
					print(f'High correlation between descriptor {descriptor_names[j]} and descriptor {descriptor_names[i]}, will omit this descriptor')
	return descriptors_to_include

#remove_highly_correlated_and_zero_features takes in the dataset splits dictionary, and does feature selection on the training set
#removes any descriptors that are only 0s for all entries and also drops all features w/|correlation coefficient| > 0.9
#Code below based partially on code associated with Dhakal & Shah (2021):
## https://github.com/ShahResearchGroup/Machine-Learning-Model-for-Imidazolium-Ionic-Liquids/blob/main/ionic_conductivity/ffann/final_model_development.py
#returns updated dataset splits dictionary after dropping highly correlated features or features that are all 0s in training set
def remove_highly_correlated_zero_features(dataset_splits_dict):
	x_data_train_df = dataset_splits_dict['x_train']
	x_data_val_df = dataset_splits_dict['x_val']
	x_data_test_df = dataset_splits_dict['x_test']

	# On training set, go through all features and drop all features with |correlation coef.| > 0.9 or features that are 0 (or have standard deviation == 0) for all entries in training set
	#Feature selection using training set
	#Remove any descriptors that are only 0s for all entries
	#Code below based on code provided with Dhakal & Shah (2021) (https://github.com/ShahResearchGroup/Machine-Learning-Model-for-Imidazolium-Ionic-Liquids/blob/main/ionic_conductivity/ffann/final_model_development.py)
	#first drop features that are 0 for all compounds
	descriptors_with_all_zeros = (x_data_train_df.columns[x_data_train_df.sum() == 0 ])
	print(f'Descriptors found to be 0 for all compounds in training set: {descriptors_with_all_zeros}')
	print(f'Will remove these descriptors from x_data\n')
	x_data_train_df.drop(columns = descriptors_with_all_zeros, inplace = True )
	x_data_val_df.drop(columns = descriptors_with_all_zeros, inplace = True )
	x_data_test_df.drop(columns = descriptors_with_all_zeros, inplace = True )

	#Check if there are any descriptors with the same value (i.e. standard deviation =  0)
	descriptors_stdevZero = (x_data_train_df.columns[x_data_train_df.std() == 0 ])
	print(f'Minimum standard deviation for descriptors in training set: {np.min(x_data_train_df.std())}')
	print(f'Descriptors found to have standard deviation = 0 across all compounds in training set: {descriptors_stdevZero}')
	print(f'Will remove these descriptors from x_data\n')
	x_data_train_df.drop(columns = descriptors_stdevZero, inplace = True )
	x_data_val_df.drop(columns = descriptors_stdevZero, inplace = True )
	x_data_test_df.drop(columns = descriptors_stdevZero, inplace = True )

	print('Number of features after removing those that were 0 for all compounds in training set or had standard deviation = 0 across all compounds in training set: ')
	print(x_data_train_df.shape[1])

	#Calculate correlation coefficient between descriptors

	#uncomment cell to calculate & plot correlation matrix
	descriptor_correlation_train_data = x_data_train_df.corr()
	#Plot correlation as heatmap
	sns.heatmap(descriptor_correlation_train_data)
	plt.title('Descriptor Correlation')
	plt.savefig(directory_to_save_results + 'descriptor_correlation_coef_matrix_train_data.png')
	plt.close()
	#https://www.geeksforgeeks.org/how-to-create-a-seaborn-correlation-heatmap-in-python/
	#https://seaborn.pydata.org/generated/seaborn.heatmap.html

	#select features that are not highly correlated w/each other 

	x_data_train_descriptors_to_keep = get_descriptors_to_keep(descriptor_correlation_train_data, print_level = 1)
	print('\n*****Descriptors to keep:*****')
	print(x_data_train_descriptors_to_keep)

	print('*****Number of features before removing those with high correlation: *****')
	print(dataset_splits_dict['x_train'].shape[1])
	print('*****Final number of features: *****')
	#update train/val/test sets to only keep features that are not highly correlated with each other
	x_data_train_df = x_data_train_df[x_data_train_descriptors_to_keep]
	x_data_val_df = x_data_val_df[x_data_train_descriptors_to_keep]
	x_data_test_df = x_data_test_df[x_data_train_descriptors_to_keep]

	dataset_splits_dict['x_train'] = x_data_train_df
	dataset_splits_dict['x_val'] = x_data_val_df
	dataset_splits_dict['x_test'] = x_data_test_df

	print(dataset_splits_dict['x_train'].shape[1])
	return dataset_splits_dict

#combine_train_and_val_sets takes in a dataset splits dictionary and returns a dictionary w/the train & validation sets combined (i.e. keys are x_train_and_val, y_train_and_val, and dataset_info_train_and_val)
#note - by default x_train_and_val and y_train_and_val in output dictionary are numpy arrays, while dataset_info_train_and_val is a pandas dataframe
#if return_df is set to True, then x_train_and_val, y_train_and_val, and dataset_info_train_and_val are all returned as pandas dataframes
def combine_train_and_val_sets(dataset_splits_dict, return_df = False):
	x_data_train_df = dataset_splits_dict['x_train']
	x_data_val_df = dataset_splits_dict['x_val']
	y_data_train_df = dataset_splits_dict['y_train']
	y_data_val_df = dataset_splits_dict['y_val']
	dataset_info_train = dataset_splits_dict['dataset_info_train']
	dataset_info_val = dataset_splits_dict['dataset_info_val']

	#First combine training & validation sets/dataframes together (since when do cross validation, will create different training/validation splits using this set)
	x_data_train_and_val_df = pd.concat([x_data_train_df, x_data_val_df], ignore_index = True)
	y_data_train_and_val_df = pd.concat([y_data_train_df, y_data_val_df], ignore_index = True)
	dataset_info_train_and_val_df = pd.concat([dataset_info_train, dataset_info_val], ignore_index = True)
	print(f'Shape of training set: \n\tx:{x_data_train_df.shape}, y:{y_data_train_df.shape}, dataset_info:{dataset_info_train.shape}')
	print(f'Shape of validation set: \n\tx:{x_data_val_df.shape}, y:{y_data_val_df.shape}, dataset_info:{dataset_info_val.shape}')
	print(f'Shape of combined train & validation set: \n\tx:{x_data_train_and_val_df.shape}, y:{y_data_train_and_val_df.shape}, dataset_info:{dataset_info_train_and_val_df.shape}')

	if(return_df):
		train_and_val_sets_dict = {}
		train_and_val_sets_dict['x_train_and_val'] = x_data_train_and_val_df
		train_and_val_sets_dict['y_train_and_val'] = y_data_train_and_val_df
		train_and_val_sets_dict['dataset_info_train_and_val'] = dataset_info_train_and_val_df
		return train_and_val_sets_dict
	else:
		#Convert x & y dataframes to numpy arrays
		x_data_train_and_val = x_data_train_and_val_df.to_numpy()
		y_data_train_and_val = y_data_train_and_val_df.to_numpy()
		print(f'Shape of combined train & validation array: \n\tx:{x_data_train_and_val.shape}, y:{y_data_train_and_val.shape}')

		train_and_val_sets_dict = {}
		train_and_val_sets_dict['x_train_and_val'] = x_data_train_and_val
		train_and_val_sets_dict['y_train_and_val'] = y_data_train_and_val
		train_and_val_sets_dict['dataset_info_train_and_val'] = dataset_info_train_and_val_df
		return train_and_val_sets_dict


#eval_model_no_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info)
#also takes in model_basename to use in saving plots/results
#loads trained model to predict the desired property (y)
#computes performance metrics on training (=combined training and validation set) set and test sets
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
def eval_model_no_log_y(train_val_test_sets_dict, model_basename):
	#Combine training and validation sets
	train_val_sets_dict = combine_train_and_val_sets(train_val_test_sets_dict, return_df = True)

	#Pull out data
	x_train_df = train_val_sets_dict['x_train_and_val']	
	x_test_df = train_val_test_sets_dict['x_test']

	y_train_df = train_val_sets_dict['y_train_and_val']
	y_test_df = train_val_test_sets_dict['y_test']

	#have x input include feature names
	x_train = x_train_df #.to_numpy()
	x_test = x_test_df #.to_numpy()

	y_train = y_train_df
	y_test = y_test_df

	dataset_info_train = train_val_sets_dict['dataset_info_train_and_val']
	dataset_info_test = train_val_test_sets_dict['dataset_info_test']

	train_test_sets_dict = {'x_train': x_train_df, 'x_test': x_test_df, 'y_train': y_train_df, 'y_test': y_test_df, 'dataset_info_train': dataset_info_train, 'dataset_info_test': dataset_info_test}

	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''

	#Load trained model
	#https://scikit-learn.org/stable/model_persistence.html
	saved_model_path = 'saved_models' + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + input_features_to_use + '_model.pkl'
	with open(saved_model_path, 'rb') as file:
		model_pipeline = pickle.load(file)
	
	#create dictionary to store values of the performance metrics
	performance_metrics_train = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}
	performance_metrics_test = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}

	#Use pipelined model, predict on test set, save performance metrics
	#with log transformation of y	
	y_train_pred_no_log_y = model_pipeline.predict(x_train)
	y_test_pred_no_log_y = model_pipeline.predict(x_test)
	
	train_performance_no_log_y = compute_score(y_true=y_train, y_pred = y_train_pred_no_log_y)
	performance_metrics_train['R2'] = float(train_performance_no_log_y["R2"])
	performance_metrics_train['MedAE'] = float(train_performance_no_log_y["MedAE"])
	performance_metrics_train['MAE'] = float(train_performance_no_log_y["MAE"])
	performance_metrics_train['MSE'] = float(train_performance_no_log_y["MSE"])

	test_performance_no_log_y = compute_score(y_true=y_test, y_pred = y_test_pred_no_log_y)
	performance_metrics_test['R2'] = float(test_performance_no_log_y["R2"])
	performance_metrics_test['MedAE'] = float(test_performance_no_log_y["MedAE"])
	performance_metrics_test['MAE'] = float(test_performance_no_log_y["MAE"])
	performance_metrics_test['MSE'] = float(test_performance_no_log_y["MSE"])
	
	#create plots w/results
	plot_basename_train = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str +  'training_and_validation_set' + '_'
	create_parity_plot(y=y_train, yhat=y_train_pred_no_log_y, model_label=model_basename,plot_filename=plot_basename_train + 'parity_plot', predicted_value_label='',  print_level= 1)

	plot_basename_test = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'test_set' + '_'
	create_parity_plot(y=y_test, yhat=y_test_pred_no_log_y, model_label=model_basename,plot_filename=plot_basename_test + 'parity_plot', predicted_value_label='',  print_level= 1)

	return [performance_metrics_train, performance_metrics_test]


#eval_model_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info)
#also takes in model_basename to use in saving plots/results
#loads trained model to predict the log(desired property) (log(y))
#computes performance metrics on training (=combined training and validation set) and test sets
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
def eval_model_log_y(train_val_test_sets_dict, model_basename):
	#Combine training and validation sets
	train_val_sets_dict = combine_train_and_val_sets(train_val_test_sets_dict, return_df = True)

	#Pull out data
	x_train_df = train_val_sets_dict['x_train_and_val']	
	x_test_df = train_val_test_sets_dict['x_test']

	y_train_df = train_val_sets_dict['y_train_and_val']
	y_test_df = train_val_test_sets_dict['y_test']

	#have x input include feature names
	x_train = x_train_df #.to_numpy()
	x_test = x_test_df #.to_numpy()

	y_train = y_train_df
	y_test = y_test_df

	dataset_info_train = train_val_sets_dict['dataset_info_train_and_val']
	dataset_info_test = train_val_test_sets_dict['dataset_info_test']

	train_test_sets_dict = {'x_train': x_train_df, 'x_test': x_test_df, 'y_train': y_train_df, 'y_test': y_test_df, 'dataset_info_train': dataset_info_train, 'dataset_info_test': dataset_info_test}

	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''

	#Load trained model
	#https://scikit-learn.org/stable/model_persistence.html
	saved_model_path = 'saved_models' + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + input_features_to_use + '_model.pkl'
	with open(saved_model_path, 'rb') as file:
		model_pipeline = pickle.load(file)

	#create dictionary to store values of the performance metrics
	performance_metrics_train = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}
	performance_metrics_test = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}

	#Use pipelined model, predict on test set, save performance metrics
	#with log transformation of y	
	y_train_pred_log_y = np.float_power(10.0,model_pipeline.predict(x_train))
	y_test_pred_log_y = np.float_power(10.0,model_pipeline.predict(x_test))
	
	train_performance_log_y = compute_score(y_true=y_train, y_pred = y_train_pred_log_y)
	performance_metrics_train['R2'] = float(train_performance_log_y["R2"])
	performance_metrics_train['MedAE'] = float(train_performance_log_y["MedAE"])
	performance_metrics_train['MAE'] = float(train_performance_log_y["MAE"])
	performance_metrics_train['MSE'] = float(train_performance_log_y["MSE"])

	test_performance_log_y = compute_score(y_true=y_test, y_pred = y_test_pred_log_y)
	performance_metrics_test['R2'] = float(test_performance_log_y["R2"])
	performance_metrics_test['MedAE'] = float(test_performance_log_y["MedAE"])
	performance_metrics_test['MAE'] = float(test_performance_log_y["MAE"])
	performance_metrics_test['MSE'] = float(test_performance_log_y["MSE"])
	
	#create plots w/results
	plot_basename_train = directory_to_save_results + '/' +  model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'training_and_validation_set' + '_'
	create_parity_plot(y=y_train, yhat=y_train_pred_log_y, model_label=model_basename,plot_filename=plot_basename_train + 'parity_plot', predicted_value_label='',  print_level= 1)

	plot_basename_test = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'test_set' + '_'
	create_parity_plot(y=y_test, yhat=y_test_pred_log_y, model_label=model_basename,plot_filename=plot_basename_test + 'parity_plot', predicted_value_label='',  print_level= 1)

	return [performance_metrics_train, performance_metrics_test]



#eval_models will call relevant functions based on arguments passed into the code to load and evaluate models
def eval_models():
	#read in dataset splits
	train_val_test_splits_dict = read_in_dataset_splits()

	#check no overlaps in train/val/test splits
	verify_no_overlaps_between_splits(train_val_test_splits_dict)

	#drop/remove any highly correlated features or features that are zero or have standard deviation=0 on training set
	train_val_test_splits_dict_removed_correlated_features = remove_highly_correlated_zero_features(train_val_test_splits_dict)

	train_and_val_sets_combined_dict = combine_train_and_val_sets(train_val_test_splits_dict_removed_correlated_features)
	
	#calculate metrics on training & test sets
	print(f'Input feature set: {input_features_to_use}')
	if(model_type == 'linear_l1'):
		if(molar_conductivity_or_ionicity == 'ionicity'):
			#no log transformation
			model_basename_str = 'l1_no_log_y' + '_opt_alpha_' + str(opt_hyperparameter_val) + '_combinedTrainAndVal'
			overall_model_performance = eval_model_no_log_y(train_val_test_splits_dict_removed_correlated_features, model_basename_str)
			print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on y, combined train and val sets: ****')
			print(overall_model_performance)
			print('\n')
		else:
			#conductivity - apply log transformation
			model_basename_str = 'l1_log_y'+ '_opt_alpha_' + str(opt_hyperparameter_val) + '_combinedTrainAndVal'
			overall_model_performance_log_y = eval_model_log_y(train_val_test_splits_dict_removed_correlated_features, model_basename_str)
			print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on log(y), combined train and val sets: ****')
			print(overall_model_performance_log_y)
			print('\n')
	elif(model_type == 'xgboost'):
		if(molar_conductivity_or_ionicity == 'ionicity'):
			#no log transformation
			opt_max_depth = opt_hyperparameter_val[0]
			opt_n_estimators = opt_hyperparameter_val[1]
			model_basename_str = 'xgb_no_log_y'+ '_opt_max_depth_' + str(opt_max_depth) + '_opt_n_estimators_' +  str(opt_n_estimators) + '_combinedTrainAndVal'
			overall_model_performance = eval_model_no_log_y(train_val_test_splits_dict_removed_correlated_features, model_basename_str)
			print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost model trained on y, combined train and val sets: ****')
			print(overall_model_performance)
			print('\n')
		else:
			#conductivity - apply log transformation
			opt_max_depth = opt_hyperparameter_val[0]
			opt_n_estimators = opt_hyperparameter_val[1]
			model_basename_str = 'xgb_log_y'+ '_opt_max_depth_' + str(opt_max_depth) + '_opt_n_estimators_' +  str(opt_n_estimators) + '_combinedTrainAndVal'
			overall_model_performance_log_y = eval_model_log_y(train_val_test_splits_dict_removed_correlated_features, model_basename_str)
			print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost trained on log(y), combined train and val sets: ****')
			print(overall_model_performance_log_y)
			print('\n')
	else:
		sys.exit(f'Invalid model_type provided (input was : {model_type}, options are: linear_l1, xgboost)')

#Run code/evaluate models
eval_models()