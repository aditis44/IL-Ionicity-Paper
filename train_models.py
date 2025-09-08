#Code used to train models to predict ionicity or molar conductivity, and tune hyperparameters
#Uses preprocessed dataset created using data from NIST ILThermo database (https://ilthermo.boulder.nist.gov, Kazakov et al., Dong et al. (2007)) & ILThermoPy package (https://github.com/IvanChernyshov/ILThermoPy)

#Uses cross validation, trains specified model to predict conductivity or ionicity, loops over multiple hyperparameter values
#Model types tested/evaluated: linear model w/L1 regularization and XGBoost
#Code also provides results for a dummy regressor that always predicts the mean

#If optimal hyperparameter values are passed into the model, then code will train models on the full training set & report results on the test set
#Code will also compare results to a dummy regressor

#Hyperparameters varied: 
## Linear models - regularization strength/alpha
## XGBoost models - max_depth values & n_estimators (fixes/uses learning_rate=0.07, colsample_bytree=0.8, subsample=0.4, colsample_bylevel=0.1)

#When predicting conductivity, a log transformation of y values is applied (found to improve model performance) (no log transformation when predicting ionicity)
#Before training models, downselect features by removing features that are highly correlated with each other (|correlation coef| > 0.9)
## Input to models: features that are not highly correlated with each other 

#Code uses Weights & Biases (https://wandb.ai/site/) to track results


#Imports
import wandb #import weights & biases for logging results

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn import pipeline

from sklearn.linear_model import  Lasso

from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor

import xgboost as xgb

import pandas as pd
import os
import copy
import argparse

import createPlotsML
from createPlotsML import *

import feature_importance_analysis
from feature_importance_analysis import *

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
parser = argparse.ArgumentParser(description="information on which dataset to train/evaluate models on, what path to save results to, and what model type to train")
parser.add_argument('-molar_conductivity_or_ionicity', type=str, dest='molar_conductivity_or_ionicity', help='value to predict (y data to train on) is either conductivity or ionicity')
parser.add_argument('-ionic_radii_estimation_method', type=str, dest='ionic_radii_estimation_method', help='Ionic radii estimation method, options: NA, RDKitDCLV_vdWVolAndSA')
parser.add_argument('-input_features_to_use', type=str, dest='input_features_to_use',help='Set of input features to use (x values) for training the ML model, options: RDKit, RDKit_and_sigma_profiles, sigma_profiles, S_i, RDKit_and_S_i, moment_waps_wans, RDKit_and_moment_waps_wans_desc, S_i_and_moment_waps_wans')
parser.add_argument('-directory_to_save_results', type=str, dest='directory_to_save_results', help='directory where results (csv & plots) will be saved to')
parser.add_argument('-model_type', type=str, dest='model_type', help='type of model to train (options: linear_l1, xgboost)')
parser.add_argument('-compare_to_dummy_regressor', type=str, dest='compare_to_dummy_regressor', help='whether to also train dummy regressor & save results')
parser.add_argument('-do_cross_validation', type=str, dest='do_cross_validation', help='whether to do cross validation or train models on full training set, provide results on validation & test sets')
parser.add_argument('-opt_hyperparameter_val', type=float, dest='opt_hyperparameter_val', help='optimal hyperparameter (alpha or max_depth) value to use when training model on full training set, will provide results on validation & test sets')
parser.add_argument('-opt_hyperparameter_val2', type=int, dest='opt_hyperparameter_val2', help='second optimal hyperparameter (n_estimators) value to use when training RF or XGBoost model on full training set, will provide results on validation & test sets')


args = parser.parse_args()
molar_conductivity_or_ionicity = args.molar_conductivity_or_ionicity #whether to predict molar conductivity or predict ionicity
ionic_radii_estimation_method = args.ionic_radii_estimation_method
input_features_to_use = args.input_features_to_use
directory_to_save_results = args.directory_to_save_results
model_type = args.model_type
compare_to_dummy_regressor =args.compare_to_dummy_regressor
do_cross_validation = args.do_cross_validation
if(do_cross_validation != 'True'):
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



#### Scale features, train models, do cross-validation ####
# Use Pipelines in sklearn to help ensure preprocessing is consistent, and avoiding data leakage
# References:
# - https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.make_pipeline.html
# - https://scikit-learn.org/stable/modules/compose.html
# - https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html
# - https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
# - https://scikit-learn.org/stable/common_pitfalls.html
# - https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
# - https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_refit_callable.html
# - https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html


#decomposeModelPerformanceByCategory takes in 2 arrays: y_curr_fold and yhat_curr_fold, along with 1 dataframe: dataset_info_curr_fold_df, that contains information on the IL represented in each row/entry for y & yhat
#also takes two booleans: cationFamily and longestAlkylChain (default - both False, note: only 1 can be True at a time) - based on which is true, the model performance will be provided for each value of either cationFamily or longestAlkylChain
#returns a nested dictionary w/model results by cation family or longest alkyl chain length (first set of keys = R2, MedAE, MAE, MSE; second set of keys: unique values for cation family or alkyl chain length based on the provided dataset_info)
def decomposeModelPerformanceByCategory(y_curr_fold, yhat_curr_fold, dataset_info_curr_fold_df, cationFamily=False, longestAlkylChain=False):
	compiled_dataframe = dataset_info_curr_fold_df.copy()
	compiled_dataframe['y_true'] = y_curr_fold
	compiled_dataframe['y_pred'] = yhat_curr_fold
	categorized_model_performance_dict = {'R2': {}, 'MedAE': {}, 'MAE': {}, 'MSE': {}}
	if(cationFamily and longestAlkylChain):
		sys.exit('Incorrect input, code requires that exactly/only one of (cationFamily, longestAlkylChain) are True, however both were set as True in input, need to edit & rerun')
	elif(cationFamily):
		cation_families = list(set(dataset_info_curr_fold_df['Cation Family']))
		for curr_cation_family in cation_families:
			compiled_dataframe_curr_cation_family = compiled_dataframe[compiled_dataframe['Cation Family'] == curr_cation_family]
			y_true_curr_cation_family = (compiled_dataframe_curr_cation_family['y_true']).to_numpy()
			y_pred_curr_cation_family = (compiled_dataframe_curr_cation_family['y_pred']).to_numpy()
			if(np.size(y_true_curr_cation_family) >=2):
				#need at least 2 entries to compute R^2
				performance_curr_cation_family= compute_score(y_true=y_true_curr_cation_family, y_pred = y_pred_curr_cation_family)
				categorized_model_performance_dict['R2'][curr_cation_family] = float(performance_curr_cation_family["R2"])
				categorized_model_performance_dict['MedAE'][curr_cation_family] = float(performance_curr_cation_family["MedAE"])
				categorized_model_performance_dict['MAE'][curr_cation_family] = float(performance_curr_cation_family["MAE"])
				categorized_model_performance_dict['MSE'][curr_cation_family] = float(performance_curr_cation_family["MSE"])
	elif(longestAlkylChain):
		alkyl_chain_lengths = list(set(dataset_info_curr_fold_df['Longest Alkyl Chain']))
		for curr_chain_length in alkyl_chain_lengths:
			compiled_dataframe_curr_chain_length= compiled_dataframe[compiled_dataframe['Longest Alkyl Chain'] == curr_chain_length]
			y_true_curr_chain_length = (compiled_dataframe_curr_chain_length['y_true']).to_numpy()
			y_pred_curr_chain_length = (compiled_dataframe_curr_chain_length['y_pred']).to_numpy()
			if(np.size(y_true_curr_chain_length) >=2):
				#need at least 2 entries to compute R^2
				performance_curr_chain_length= compute_score(y_true=y_true_curr_chain_length, y_pred = y_pred_curr_chain_length)
				categorized_model_performance_dict['R2'][curr_chain_length] = float(performance_curr_chain_length["R2"])
				categorized_model_performance_dict['MedAE'][curr_chain_length] = float(performance_curr_chain_length["MedAE"])
				categorized_model_performance_dict['MAE'][curr_chain_length] = float(performance_curr_chain_length["MAE"])
				categorized_model_performance_dict['MSE'][curr_chain_length] = float(performance_curr_chain_length["MSE"])
	else:
		sys.exit('Incorrect input, code requires that exactly/only one of (cationFamily, longestAlkylChain) are True, however both were False in input, need to edit & rerun')
	return categorized_model_performance_dict

#logAndCompileOverallResultsPerFold takes in a Weights & Biases run, dictionaries with performance metrics per fold, performance metrics per fold by cation family, and performance metrics per fold by alkyl chain length
#also takes in a list with all possible cation family names & alkyl chain lengths
#returns a dictionary w/overall model performance results (not broken down by cation family or alkyl chain length)
def logAndCompileOverallResultsPerFold(wandb_run, metrics_per_fold, metrics_per_fold_by_cation_family, metrics_per_fold_by_alkyl_chain_length, cation_families_list_all, alkyl_chain_lengths_list_all):
	#create dictionary to store overall values of the performance metrics
	overall_performance_metrics = {'R2': {}, 'MedAE': {}, 'MAE': {}, 'MSE': {}}
	
	#Calculate average performance metric over all k folds
	overall_performance_metrics['R2']['Average'] = np.mean(metrics_per_fold['R2'])
	overall_performance_metrics['R2']['Standard Deviation'] = np.std(metrics_per_fold['R2'])
	overall_performance_metrics['MedAE']['Average'] = np.mean(metrics_per_fold['MedAE'])
	overall_performance_metrics['MedAE']['Standard Deviation'] = np.std(metrics_per_fold['MedAE'])
	overall_performance_metrics['MAE']['Average'] = np.mean(metrics_per_fold['MAE'])
	overall_performance_metrics['MAE']['Standard Deviation'] = np.std(metrics_per_fold['MAE'])
	overall_performance_metrics['MSE']['Average'] = np.mean(metrics_per_fold['MSE'])
	overall_performance_metrics['MSE']['Standard Deviation'] = np.std(metrics_per_fold['MSE'])
	
	#log overall results in W&B (only log results when comparing y & yhat, since log(y) vs log(yhat) comparison is not as useful for determining model performance & already generate plots/save csv files for those folds
	wandb_run.summary['Average R^2 over 5 folds, validation set'] = overall_performance_metrics['R2']['Average']
	wandb_run.summary['Standard Deviation R^2 over 5 folds, validation set'] = overall_performance_metrics['R2']['Standard Deviation']
	wandb_run.summary['Average MedAE over 5 folds, validation set'] = overall_performance_metrics['MedAE']['Average']	
	wandb_run.summary['Standard Deviation MedAE over 5 folds, validation set'] = overall_performance_metrics['MedAE']['Standard Deviation']
	wandb_run.summary['Average MAE over 5 folds, validation set'] = overall_performance_metrics['MAE']['Average']	
	wandb_run.summary['Standard Deviation MAE over 5 folds, validation set'] = overall_performance_metrics['MAE']['Standard Deviation']
	wandb_run.summary['Average MSE over 5 folds, validation set'] = overall_performance_metrics['MSE']['Average']	
	wandb_run.summary['Standard Deviation MSE over 5 folds, validation set'] = overall_performance_metrics['MSE']['Standard Deviation']

	for curr_cation_family in cation_families_list_all:
		curr_family_r2 = metrics_per_fold_by_cation_family['R2'][curr_cation_family]
		if(len(curr_family_r2) >= 2):
			#need the cation family to be present in validation set of at least 2 folds to get statistics
			curr_family_medAE = metrics_per_fold_by_cation_family['MedAE'][curr_cation_family]
			curr_family_mae = metrics_per_fold_by_cation_family['MAE'][curr_cation_family]
			curr_family_mse = metrics_per_fold_by_cation_family['MSE'][curr_cation_family]
			wandb_run.summary[f'Average R^2 over 5 folds, {curr_cation_family} ILs, validation set'] = np.mean(curr_family_r2)
			wandb_run.summary[f'Standard Deviation R^2 over 5 folds, {curr_cation_family} ILs, validation set'] = np.std(curr_family_r2)
			wandb_run.summary[f'Average MedAE over 5 folds, {curr_cation_family} ILs, validation set'] = np.mean(curr_family_medAE)
			wandb_run.summary[f'Standard Deviation MedAE over 5 folds, {curr_cation_family} ILs, validation set'] = np.std(curr_family_medAE)
			wandb_run.summary[f'Average MAE over 5 folds, {curr_cation_family} ILs, validation set'] = np.mean(curr_family_mae)
			wandb_run.summary[f'Standard Deviation MAE over 5 folds, {curr_cation_family} ILs, validation set'] = np.std(curr_family_mae)
			wandb_run.summary[f'Average MSE over 5 folds, {curr_cation_family} ILs, validation set'] = np.mean(curr_family_mse)
			wandb_run.summary[f'Standard Deviation MSE over 5 folds, {curr_cation_family} ILs, validation set'] = np.std(curr_family_mse)
		else:
			print(f'{curr_cation_family} was not present in at least 2 folds, so cannot compute statistics for this cation family')

	for curr_chain_length in alkyl_chain_lengths_list_all:
		curr_chain_length_r2 = metrics_per_fold_by_alkyl_chain_length['R2'][curr_chain_length]
		if(len(curr_chain_length_r2) >= 2):
			#need the alkyl chain length to be present in validation set of at least 2 folds to get statistics
			curr_chain_length_medAE = metrics_per_fold_by_alkyl_chain_length['MedAE'][curr_chain_length]
			curr_chain_length_mae = metrics_per_fold_by_alkyl_chain_length['MAE'][curr_chain_length]
			curr_chain_length_mse = metrics_per_fold_by_alkyl_chain_length['MSE'][curr_chain_length]
			wandb_run.summary[f'Average R^2 over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.mean(curr_chain_length_r2)
			wandb_run.summary[f'Standard Deviation R^2 over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.std(curr_chain_length_r2)
			wandb_run.summary[f'Average MedAE over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.mean(curr_chain_length_medAE)
			wandb_run.summary[f'Standard Deviation MedAE over 5 folds, {curr_chain_length}C longest alkyl chain, validation set'] = np.std(curr_chain_length_medAE)
			wandb_run.summary[f'Average MAE over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.mean(curr_chain_length_mae)
			wandb_run.summary[f'Standard Deviation MAE over 5 folds, {curr_chain_length}C longest alkyl chain, validation set'] = np.std(curr_chain_length_mae)
			wandb_run.summary[f'Average MSE over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.mean(curr_chain_length_mse)
			wandb_run.summary[f'Standard Deviation MSE over 5 folds, ILs w/{curr_chain_length}C longest alkyl chain, validation set'] = np.std(curr_chain_length_mse)
		else:
			print(f'{curr_chain_length} was not present in at least 2 folds, so cannot compute statistics for this alkyl chain length')
	return overall_performance_metrics

#logOverallResults_trainAndValCombined takes in a Weights & Biases run, lists with performance metric dictionaries for the train/test sets (where train set = training + validation set), performance metrics for each set by cation family, and performance metrics for each set by alkyl chain length
#NOTE: assumed order in each list w/performance metrics is [train, test]
def logOverallResults_trainAndValCombined(wandb_run, metrics_list, metrics_by_cation_family_list, metrics_by_alkyl_chain_length_list):
	#create dictionary to store overall values of the performance metrics
	overall_performance_metrics_train = metrics_list[0]
	overall_performance_metrics_test = metrics_list[1]

	#log overall results in W&B (only log results when comparing y & yhat, since log(y) vs log(yhat) comparison is not as useful for determining model performance & already generate plots/save csv files for those folds
	wandb_run.summary['R^2, train set'] = overall_performance_metrics_train['R2']
	wandb_run.summary['MedAE, train set'] = overall_performance_metrics_train['MedAE']
	wandb_run.summary['MAE, train set'] = overall_performance_metrics_train['MAE']	
	wandb_run.summary['MSE, train set'] = overall_performance_metrics_train['MSE']

	wandb_run.summary['R^2, test set'] = overall_performance_metrics_test['R2']
	wandb_run.summary['MedAE, test set'] = overall_performance_metrics_test['MedAE']
	wandb_run.summary['MAE, test set'] = overall_performance_metrics_test['MAE']	
	wandb_run.summary['MSE, test set'] = overall_performance_metrics_test['MSE']

	for i in range(2):
		if(i == 0):
			set_str = 'train set'
		else:
			set_str = 'test set'

		metrics_by_cation_family = metrics_by_cation_family_list[i]
		cation_families_list_all = list(metrics_by_cation_family['R2'].keys())
		metrics_by_alkyl_chain_length = metrics_by_alkyl_chain_length_list[i]
		alkyl_chain_lengths_list_all = list(metrics_by_alkyl_chain_length['R2'].keys())
		
		for curr_cation_family in cation_families_list_all:
			curr_family_r2 = metrics_by_cation_family['R2'][curr_cation_family]
			curr_family_medAE = metrics_by_cation_family['MedAE'][curr_cation_family]
			curr_family_mae = metrics_by_cation_family['MAE'][curr_cation_family]
			curr_family_mse = metrics_by_cation_family['MSE'][curr_cation_family]
			wandb_run.summary[f'R^2, {curr_cation_family} ILs, {set_str}'] = curr_family_r2
			wandb_run.summary[f'MedAE, {curr_cation_family} ILs, {set_str}'] = curr_family_medAE
			wandb_run.summary[f'MAE, {curr_cation_family} ILs, {set_str}'] = curr_family_mae
			wandb_run.summary[f'MSE, {curr_cation_family} ILs, {set_str}'] = curr_family_mse
		
		for curr_chain_length in alkyl_chain_lengths_list_all:
			curr_chain_length_r2 = metrics_by_alkyl_chain_length['R2'][curr_chain_length]
			curr_chain_length_medAE = metrics_by_alkyl_chain_length['MedAE'][curr_chain_length]
			curr_chain_length_mae = metrics_by_alkyl_chain_length['MAE'][curr_chain_length]
			curr_chain_length_mse = metrics_by_alkyl_chain_length['MSE'][curr_chain_length]

			wandb_run.summary[f'R^2, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_r2
			wandb_run.summary[f'MedAE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_medAE
			wandb_run.summary[f'MAE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_mae
			wandb_run.summary[f'MSE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_mse
	


#logOverallResults takes in a Weights & Biases run, lists with performance metric dictionaries for the train/validation/test sets, performance metrics for each set by cation family, and performance metrics for each set by alkyl chain length
#NOTE: assumed order in each list w/performance metrics is [train, val, test]
def logOverallResults(wandb_run, metrics_list, metrics_by_cation_family_list, metrics_by_alkyl_chain_length_list):
	#create dictionary to store overall values of the performance metrics
	overall_performance_metrics_train = metrics_list[0]
	overall_performance_metrics_val = metrics_list[1]
	overall_performance_metrics_test = metrics_list[2]

	#log overall results in W&B (only log results when comparing y & yhat, since log(y) vs log(yhat) comparison is not as useful for determining model performance & already generate plots/save csv files for those folds
	wandb_run.summary['R^2, train set'] = overall_performance_metrics_train['R2']
	wandb_run.summary['MedAE, train set'] = overall_performance_metrics_train['MedAE']
	wandb_run.summary['MAE, train set'] = overall_performance_metrics_train['MAE']	
	wandb_run.summary['MSE, train set'] = overall_performance_metrics_train['MSE']

	wandb_run.summary['R^2, validation set'] = overall_performance_metrics_val['R2']
	wandb_run.summary['MedAE, validation set'] = overall_performance_metrics_val['MedAE']
	wandb_run.summary['MAE, validation set'] = overall_performance_metrics_val['MAE']	
	wandb_run.summary['MSE, validation set'] = overall_performance_metrics_val['MSE']

	wandb_run.summary['R^2, test set'] = overall_performance_metrics_test['R2']
	wandb_run.summary['MedAE, test set'] = overall_performance_metrics_test['MedAE']
	wandb_run.summary['MAE, test set'] = overall_performance_metrics_test['MAE']	
	wandb_run.summary['MSE, test set'] = overall_performance_metrics_test['MSE']

	for i in range(3):
		if(i == 0):
			set_str = 'train set'
		elif(i == 1):
			set_str = 'validation set'
		else:
			set_str = 'test set'

		metrics_by_cation_family = metrics_by_cation_family_list[i]
		cation_families_list_all = list(metrics_by_cation_family['R2'].keys())
		metrics_by_alkyl_chain_length = metrics_by_alkyl_chain_length_list[i]
		alkyl_chain_lengths_list_all = list(metrics_by_alkyl_chain_length['R2'].keys())
		
		for curr_cation_family in cation_families_list_all:
			curr_family_r2 = metrics_by_cation_family['R2'][curr_cation_family]
			curr_family_medAE = metrics_by_cation_family['MedAE'][curr_cation_family]
			curr_family_mae = metrics_by_cation_family['MAE'][curr_cation_family]
			curr_family_mse = metrics_by_cation_family['MSE'][curr_cation_family]
			wandb_run.summary[f'R^2, {curr_cation_family} ILs, {set_str}'] = curr_family_r2
			wandb_run.summary[f'MedAE, {curr_cation_family} ILs, {set_str}'] = curr_family_medAE
			wandb_run.summary[f'MAE, {curr_cation_family} ILs, {set_str}'] = curr_family_mae
			wandb_run.summary[f'MSE, {curr_cation_family} ILs, {set_str}'] = curr_family_mse
		
		for curr_chain_length in alkyl_chain_lengths_list_all:
			curr_chain_length_r2 = metrics_by_alkyl_chain_length['R2'][curr_chain_length]
			curr_chain_length_medAE = metrics_by_alkyl_chain_length['MedAE'][curr_chain_length]
			curr_chain_length_mae = metrics_by_alkyl_chain_length['MAE'][curr_chain_length]
			curr_chain_length_mse = metrics_by_alkyl_chain_length['MSE'][curr_chain_length]

			wandb_run.summary[f'R^2, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_r2
			wandb_run.summary[f'MedAE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_medAE
			wandb_run.summary[f'MAE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_mae
			wandb_run.summary[f'MSE, ILs w/{curr_chain_length}C longest alkyl chain, {set_str}'] = curr_chain_length_mse
	

#train_model_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info), sklearn model pipeline, model configuration dictionary to save to Weights & Biases
#also takes in model_basename to use in saving plots/results
#trains model to predict the log(desired property) (log(y))
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
#Approach:
#	- for each of the k folds:
# 		- train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#	- compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_model_log_y_cv(train_and_val_sets_dict, model_pipeline, model_configs, model_basename):
	if(molar_conductivity_or_ionicity == 'ionicity'): #predicting ionicity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsIonicity', entity='aseshad4', config = model_configs)
	else: #predicting conductivity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsMolarConductivity', entity='aseshad4', config = model_configs)
	
	#Pull out data
	x_data_train_and_val = train_and_val_sets_dict['x_train_and_val']
	y_data_train_and_val = train_and_val_sets_dict['y_train_and_val']
	dataset_info_train_and_val_df = train_and_val_sets_dict['dataset_info_train_and_val']

	#Now create pipeline for scaling the features & do cross validation
	#groups for k-fold cross validation
	groups_data = dataset_info_train_and_val_df['Compound']

	#create dictionary to store values of the performance metrics for each k fold
	performance_metrics_per_fold = {'R2': [], 'MedAE': [], 'MAE': [], 'MSE': []}
	
	#create dictionary to store values of performance metrics for each k fold by cation family & alkyl chain length
	cation_families_all = list(set(dataset_info_train_and_val_df['Cation Family']))
	alkyl_chain_lengths_all = list(set(dataset_info_train_and_val_df['Longest Alkyl Chain']))
	cation_family_dict = {}
	alkyl_chain_lengths_dict = {}
	for cation_family in cation_families_all:
		cation_family_dict[cation_family] = []
	for alkyl_chain_length in alkyl_chain_lengths_all:
		alkyl_chain_lengths_dict[alkyl_chain_length] = []
	performance_metrics_by_cation_family_per_fold = {'R2': copy.deepcopy(cation_family_dict), 'MedAE': copy.deepcopy(cation_family_dict), 'MAE': copy.deepcopy(cation_family_dict), 'MSE': copy.deepcopy(cation_family_dict)}
	performance_metrics_by_alkyl_chain_length_per_fold = {'R2': copy.deepcopy(alkyl_chain_lengths_dict), 'MedAE': copy.deepcopy(alkyl_chain_lengths_dict), 'MAE': copy.deepcopy(alkyl_chain_lengths_dict), 'MSE': copy.deepcopy(alkyl_chain_lengths_dict)}
	#create separate dictionaries for each performance metric, create by making a deep copy of the performance_metrics_base_dict
	##https://www.geeksforgeeks.org/deep-copy-of-a-dictionary-in-python/
	
	#Train models, do cross validation
	group_kfold_cv = GroupKFold(n_splits = 5) #paper did 5-fold CV
	y_val_kfold = []
	y_preds_kfold = []
	for i, (train_index, val_index) in enumerate(group_kfold_cv.split(x_data_train_and_val,y_data_train_and_val, groups = groups_data)):
		print(f'Fold {i+1}')
		#Split data for current fold
		print(f'Train indices: {train_index}, \n Validation indices: {val_index}')
		x_train = x_data_train_and_val[train_index]
		y_train = y_data_train_and_val[train_index]
		
		x_val = x_data_train_and_val[val_index]
		y_val = y_data_train_and_val[val_index]
		dataset_info_val_curr_fold_df = dataset_info_train_and_val_df.iloc[val_index]
		
		print(f'Size of fold {i+1} train set: {len(train_index)}; x_train shape: {np.shape(x_train)}; y_train shape: {np.shape(y_train)}')
		print(f'Size of fold {i+1} val set: {len(val_index)}; x_val shape: {np.shape(x_val)}; y_val shape: {np.shape(y_val)}, dataset_info shape: {dataset_info_val_curr_fold_df.shape}')
		
		#Use pipelined model, predict on validation set, save performance metrics
		#with log transformation of y
		model_pipeline.fit(x_train, np.log10(y_train))
		
		y_val_pred_log_y = np.float_power(10.0,model_pipeline.predict(x_val))
		y_preds_kfold.append(y_val_pred_log_y)
		y_val_kfold.append(y_val)

		performance_log_y = compute_score(y_true=y_val, y_pred = y_val_pred_log_y)
		performance_metrics_per_fold['R2'].append(float(performance_log_y["R2"]))
		performance_metrics_per_fold['MedAE'].append(float(performance_log_y["MedAE"]))
		performance_metrics_per_fold['MAE'].append(float(performance_log_y["MAE"]))
		performance_metrics_per_fold['MSE'].append(float(performance_log_y["MSE"]))
		
		#also determine metrics by cation family & alkyl chain length (just compare y & yhat, don't add in comparison of log(y) vs log(yhat))
		performance_metrics_by_cation_family = decomposeModelPerformanceByCategory(y_curr_fold = y_val, yhat_curr_fold = y_val_pred_log_y, dataset_info_curr_fold_df=dataset_info_val_curr_fold_df, cationFamily=True)
		performance_metrics_by_alkyl_chain_length = decomposeModelPerformanceByCategory(y_curr_fold = y_val, yhat_curr_fold = y_val_pred_log_y, dataset_info_curr_fold_df=dataset_info_val_curr_fold_df, longestAlkylChain=True)
		cation_family_keys = list(performance_metrics_by_cation_family['R2'].keys())
		alkyl_chain_length_keys = list(performance_metrics_by_alkyl_chain_length['R2'].keys())
		for cation_family_key in cation_family_keys:
			print(f'Fold {i}')
			performance_metrics_by_cation_family_per_fold['R2'][cation_family_key].append(performance_metrics_by_cation_family['R2'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MedAE'][cation_family_key].append(performance_metrics_by_cation_family['MedAE'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MAE'][cation_family_key].append(performance_metrics_by_cation_family['MAE'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MSE'][cation_family_key].append(performance_metrics_by_cation_family['MSE'][cation_family_key])
		for alkyl_chain_length_key in alkyl_chain_length_keys:
			performance_metrics_by_alkyl_chain_length_per_fold['R2'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['R2'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MedAE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MedAE'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MAE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MAE'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MSE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MSE'][alkyl_chain_length_key])

		#Record results for current fold in W&B
		wandb.log({
			'fold': i+1,
			'validation_r2': performance_metrics_per_fold['R2'][i],
			'validation_medAE': performance_metrics_per_fold['MedAE'][i],
			'validation_mae': performance_metrics_per_fold['MAE'][i],
			'validation_mse': performance_metrics_per_fold['MSE'][i],
		})		

	#Save plot w/results
	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''
	plot_basename_kfold = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str  + 'parity_plot'
	create_parity_plot_kfolds(y=y_val_kfold, yhat=y_preds_kfold, model_label=model_basename,plot_filename=plot_basename_kfold, predicted_value_label='', print_level= 1)

	#log results in W&B over all folds & create compiled dictionary w/performance metrics
	overall_performance_metrics = logAndCompileOverallResultsPerFold(wandb_run=run, metrics_per_fold=performance_metrics_per_fold, metrics_per_fold_by_cation_family=performance_metrics_by_cation_family_per_fold, metrics_per_fold_by_alkyl_chain_length=performance_metrics_by_alkyl_chain_length_per_fold, cation_families_list_all=cation_families_all, alkyl_chain_lengths_list_all=alkyl_chain_lengths_all)
	
	#Stop W&B run
	run.finish()
	return overall_performance_metrics

#train_model_no_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info), sklearn model pipeline, model configuration dictionary to save to Weights & Biases
#also takes in model_basename to use in saving plots/results
#trains model to predict the desired property (y)
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
#Approach:
#	- for each of the k folds:
# 		- train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#	- compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_model_no_log_y_cv(train_and_val_sets_dict, model_pipeline, model_configs, model_basename):
	if(molar_conductivity_or_ionicity == 'ionicity'): #predicting ionicity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsIonicity', entity='aseshad4', config = model_configs)
	else: #predicting conductivity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsMolarConductivity', entity='aseshad4', config = model_configs)
	
	#Pull out data
	x_data_train_and_val = train_and_val_sets_dict['x_train_and_val']
	y_data_train_and_val = train_and_val_sets_dict['y_train_and_val']
	dataset_info_train_and_val_df = train_and_val_sets_dict['dataset_info_train_and_val']

	#Now create pipeline for scaling the features & do cross validation
	#groups for k-fold cross validation
	groups_data = dataset_info_train_and_val_df['Compound']

	#create dictionary to store values of the performance metrics for each k fold
	performance_metrics_per_fold = {'R2': [], 'MedAE': [], 'MAE': [], 'MSE': []}
	
	#create dictionary to store values of performance metrics for each k fold by cation family & alkyl chain length
	cation_families_all = list(set(dataset_info_train_and_val_df['Cation Family']))
	alkyl_chain_lengths_all = list(set(dataset_info_train_and_val_df['Longest Alkyl Chain']))
	cation_family_dict = {}
	alkyl_chain_lengths_dict = {}
	for cation_family in cation_families_all:
		cation_family_dict[cation_family] = []
	for alkyl_chain_length in alkyl_chain_lengths_all:
		alkyl_chain_lengths_dict[alkyl_chain_length] = []
	performance_metrics_by_cation_family_per_fold = {'R2': copy.deepcopy(cation_family_dict), 'MedAE': copy.deepcopy(cation_family_dict), 'MAE': copy.deepcopy(cation_family_dict), 'MSE': copy.deepcopy(cation_family_dict)}
	performance_metrics_by_alkyl_chain_length_per_fold = {'R2': copy.deepcopy(alkyl_chain_lengths_dict), 'MedAE': copy.deepcopy(alkyl_chain_lengths_dict), 'MAE': copy.deepcopy(alkyl_chain_lengths_dict), 'MSE': copy.deepcopy(alkyl_chain_lengths_dict)}
	#create separate dictionaries for each performance metric, create by making a deep copy of the performance_metrics_base_dict
	##https://www.geeksforgeeks.org/deep-copy-of-a-dictionary-in-python/
	
	#Train models, do cross validation
	group_kfold_cv = GroupKFold(n_splits = 5) #paper did 5-fold CV
	y_val_kfold = []
	y_vals_omit_zero_and_neg_yhat = []
	y_preds_kfold = []
	y_preds_no_log_y_omit_zero_and_neg_yhat = []
	for i, (train_index, val_index) in enumerate(group_kfold_cv.split(x_data_train_and_val,y_data_train_and_val, groups = groups_data)):
		print(f'Fold {i+1}')
		#Split data for current fold
		print(f'Train indices: {train_index}, \n Validation indices: {val_index}')
		x_train = x_data_train_and_val[train_index]
		y_train = y_data_train_and_val[train_index]
		
		x_val = x_data_train_and_val[val_index]
		y_val = y_data_train_and_val[val_index]
		dataset_info_val_curr_fold_df = dataset_info_train_and_val_df.iloc[val_index]
		
		print(f'Size of fold {i+1} train set: {len(train_index)}; x_train shape: {np.shape(x_train)}; y_train shape: {np.shape(y_train)}')
		print(f'Size of fold {i+1} val set: {len(val_index)}; x_val shape: {np.shape(x_val)}; y_val shape: {np.shape(y_val)}, dataset_info shape: {dataset_info_val_curr_fold_df.shape}')
		
		#Use pipelined model, predict on validation set, save performance metrics
		#no log transformation of y
		model_pipeline.fit(x_train, y_train)
		
		y_val_pred_no_log_y = model_pipeline.predict(x_val)
		y_preds_kfold.append(y_val_pred_no_log_y)
		y_val_kfold.append(y_val)

		#here, it is possible that the model may predict some zero or negative ionicity values --> cannot take log of this so will remove those points from y & yhat before taking the log and comparing the results
		indices_zero_and_neg_y_val_pred_no_log_y = np.argwhere(y_val_pred_no_log_y <= 0)
		y_val_pred_no_log_y_omit_zero_and_neg_yhat = np.delete(y_val_pred_no_log_y, indices_zero_and_neg_y_val_pred_no_log_y)
		y_val_omit_zero_and_neg_yhat = np.delete(y_val, indices_zero_and_neg_y_val_pred_no_log_y)
		y_preds_no_log_y_omit_zero_and_neg_yhat.append(y_val_pred_no_log_y_omit_zero_and_neg_yhat)
		y_vals_omit_zero_and_neg_yhat.append(y_val_omit_zero_and_neg_yhat)
		
		performance_no_log_y = compute_score(y_true=y_val, y_pred = y_val_pred_no_log_y)
		performance_metrics_per_fold['R2'].append(float(performance_no_log_y["R2"]))
		performance_metrics_per_fold['MedAE'].append(float(performance_no_log_y["MedAE"]))
		performance_metrics_per_fold['MAE'].append(float(performance_no_log_y["MAE"]))
		performance_metrics_per_fold['MSE'].append(float(performance_no_log_y["MSE"]))
		
		#also determine metrics by cation family & alkyl chain length (just compare y & yhat, don't add in comparison of log(y) vs log(yhat))
		performance_metrics_by_cation_family = decomposeModelPerformanceByCategory(y_curr_fold = y_val, yhat_curr_fold = y_val_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_val_curr_fold_df, cationFamily=True)
		performance_metrics_by_alkyl_chain_length = decomposeModelPerformanceByCategory(y_curr_fold = y_val, yhat_curr_fold = y_val_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_val_curr_fold_df, longestAlkylChain=True)
		cation_family_keys = list(performance_metrics_by_cation_family['R2'].keys())
		alkyl_chain_length_keys = list(performance_metrics_by_alkyl_chain_length['R2'].keys())
		for cation_family_key in cation_family_keys:
			performance_metrics_by_cation_family_per_fold['R2'][cation_family_key].append(performance_metrics_by_cation_family['R2'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MedAE'][cation_family_key].append(performance_metrics_by_cation_family['MedAE'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MAE'][cation_family_key].append(performance_metrics_by_cation_family['MAE'][cation_family_key])
			performance_metrics_by_cation_family_per_fold['MSE'][cation_family_key].append(performance_metrics_by_cation_family['MSE'][cation_family_key])
		for alkyl_chain_length_key in alkyl_chain_length_keys:
			performance_metrics_by_alkyl_chain_length_per_fold['R2'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['R2'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MedAE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MedAE'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MAE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MAE'][alkyl_chain_length_key])
			performance_metrics_by_alkyl_chain_length_per_fold['MSE'][alkyl_chain_length_key].append(performance_metrics_by_alkyl_chain_length['MSE'][alkyl_chain_length_key])

		#Record results for current fold in W&B
		wandb.log({
			'fold': i+1,
			'validation_r2': performance_metrics_per_fold['R2'][i],
			'validation_medAE': performance_metrics_per_fold['MedAE'][i],
			'validation_mae': performance_metrics_per_fold['MAE'][i],
			'validation_mse': performance_metrics_per_fold['MSE'][i],
		})	

	#Save plot w/results
	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''
	plot_basename_kfold = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'parity_plot'
	create_parity_plot_kfolds(y=y_val_kfold, yhat=y_preds_kfold, model_label=model_basename,plot_filename=plot_basename_kfold, predicted_value_label='', print_level= 1)

	#log results in W&B over all folds & create compiled dictionary w/performance metrics
	overall_performance_metrics = logAndCompileOverallResultsPerFold(wandb_run=run, metrics_per_fold=performance_metrics_per_fold, metrics_per_fold_by_cation_family=performance_metrics_by_cation_family_per_fold, metrics_per_fold_by_alkyl_chain_length=performance_metrics_by_alkyl_chain_length_per_fold, cation_families_list_all=cation_families_all, alkyl_chain_lengths_list_all=alkyl_chain_lengths_all)
	
	#Stop W&B run
	run.finish()
	return overall_performance_metrics



#train_model_on_trainAndVal_no_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info), sklearn model pipeline, model configuration dictionary to save to Weights & Biases
#also takes in model_basename to use in saving plots/results
#trains model to predict the desired property (y) using the combined training and validation sets 
#computes performance metrics on training (=combined training and validation set) set and test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
def train_model_on_trainAndVal_no_log_y(train_val_test_sets_dict, model_pipeline, model_configs, model_basename):
	if(molar_conductivity_or_ionicity == 'ionicity'): #predicting ionicity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsIonicity', entity='aseshad4', config = model_configs)
	else: #predicting conductivity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsMolarConductivity', entity='aseshad4', config = model_configs)
		
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

	#Now use pipeline for scaling the features train and evaluate the models
	#set output type for pipeline to be a pandas dataframe so feature names are saved
	#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
	model_pipeline.set_output(transform='pandas')

	#create dictionary to store values of the performance metrics
	performance_metrics_train = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}
	performance_metrics_test = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}

	#Train models, compute performance metrics
	#Use pipelined model, predict on test set, save performance metrics
	#with log transformation of y
	model_pipeline.fit(x_train, y_train)
	
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
	
	#also determine metrics by cation family & alkyl chain length (just compare y & yhat, don't add in comparison of log(y) vs log(yhat))
	performance_metrics_by_cation_family_train = decomposeModelPerformanceByCategory(y_curr_fold = y_train, yhat_curr_fold = y_train_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_train, cationFamily=True)
	performance_metrics_by_cation_family_test = decomposeModelPerformanceByCategory(y_curr_fold = y_test, yhat_curr_fold = y_test_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_test, cationFamily=True)

	performance_metrics_by_alkyl_chain_length_train = decomposeModelPerformanceByCategory(y_curr_fold = y_train, yhat_curr_fold = y_train_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_train, longestAlkylChain=True)
	performance_metrics_by_alkyl_chain_length_test = decomposeModelPerformanceByCategory(y_curr_fold = y_test, yhat_curr_fold = y_test_pred_no_log_y, dataset_info_curr_fold_df=dataset_info_test, longestAlkylChain=True)
	
	#create plots w/results
	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''
	
	plot_basename_train = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str +  'training_and_validation_set' + '_'
	create_parity_plot(y=y_train, yhat=y_train_pred_no_log_y, model_label=model_basename,plot_filename=plot_basename_train + 'parity_plot', predicted_value_label='',  print_level= 1)

	plot_basename_test = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'test_set' + '_'
	create_parity_plot(y=y_test, yhat=y_test_pred_no_log_y, model_label=model_basename,plot_filename=plot_basename_test + 'parity_plot', predicted_value_label='',  print_level= 1)

	#log results in W&B over all folds & create compiled dictionary w/performance metrics
	logOverallResults_trainAndValCombined(wandb_run=run, metrics_list=[performance_metrics_train, performance_metrics_test], metrics_by_cation_family_list=[performance_metrics_by_cation_family_train, performance_metrics_by_cation_family_test], metrics_by_alkyl_chain_length_list=[performance_metrics_by_alkyl_chain_length_train, performance_metrics_by_alkyl_chain_length_test])
	
	#Stop W&B run
	run.finish()

	#run feature importance analysis if linear model
	if(model_type == 'linear_l1'):
		feature_importance_analysis_trainAndValCombined(model_pipeline, train_test_sets_dict, model_basename, input_features_to_use, directory_to_save_results, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)

	return [performance_metrics_train, performance_metrics_test]


#train_model_on_trainAndVal_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info), sklearn model pipeline, model configuration dictionary to save to Weights & Biases
#also takes in model_basename to use in saving plots/results
#trains model to predict the log(desired property) (log(y)) using the combined training and validation sets 
#computes performance metrics on training (=combined training and validation set) and test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns dictionary w/overall performance metrics for the model
def train_model_on_trainAndVal_log_y(train_val_test_sets_dict, model_pipeline, model_configs, model_basename):
	if(molar_conductivity_or_ionicity == 'ionicity'): #predicting ionicity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsIonicity', entity='aseshad4', config = model_configs)
	else: #predicting conductivity
		#Start a W&B run
		run = wandb.init(project='IonicLiquidsMolarConductivity', entity='aseshad4', config = model_configs)
	
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

	#Now use pipeline for scaling the features train and evaluate the models
	#set output type for pipeline to be a pandas dataframe so feature names are saved
	#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
	model_pipeline.set_output(transform='pandas')

	#create dictionary to store values of the performance metrics
	performance_metrics_train = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}
	performance_metrics_test = {'R2': None, 'MedAE': None, 'MAE': None, 'MSE': None}

	#Train models, compute performance metrics
	#Use pipelined model, predict on test set, save performance metrics
	#with log transformation of y
	model_pipeline.fit(x_train, np.log10(y_train))
	
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
	
	#also determine metrics by cation family & alkyl chain length (just compare y & yhat, don't add in comparison of log(y) vs log(yhat))
	performance_metrics_by_cation_family_train = decomposeModelPerformanceByCategory(y_curr_fold = y_train, yhat_curr_fold = y_train_pred_log_y, dataset_info_curr_fold_df=dataset_info_train, cationFamily=True)
	performance_metrics_by_cation_family_test = decomposeModelPerformanceByCategory(y_curr_fold = y_test, yhat_curr_fold = y_test_pred_log_y, dataset_info_curr_fold_df=dataset_info_test, cationFamily=True)

	performance_metrics_by_alkyl_chain_length_train = decomposeModelPerformanceByCategory(y_curr_fold = y_train, yhat_curr_fold = y_train_pred_log_y, dataset_info_curr_fold_df=dataset_info_train, longestAlkylChain=True)
	performance_metrics_by_alkyl_chain_length_test = decomposeModelPerformanceByCategory(y_curr_fold = y_test, yhat_curr_fold = y_test_pred_log_y, dataset_info_curr_fold_df=dataset_info_test, longestAlkylChain=True)
	
	#create plots w/results
	if(molar_conductivity_or_ionicity == 'ionicity'):
		ionic_radii_estimation_method_str = ionic_radii_estimation_method + '_'
	else:
		ionic_radii_estimation_method_str = ''
	
	plot_basename_train = directory_to_save_results + '/' +  model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'training_and_validation_set' + '_'
	create_parity_plot(y=y_train, yhat=y_train_pred_log_y, model_label=model_basename,plot_filename=plot_basename_train + 'parity_plot', predicted_value_label='',  print_level= 1)

	plot_basename_test = directory_to_save_results + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + 'test_set' + '_'
	create_parity_plot(y=y_test, yhat=y_test_pred_log_y, model_label=model_basename,plot_filename=plot_basename_test + 'parity_plot', predicted_value_label='',  print_level= 1)

	#log results in W&B over all folds & create compiled dictionary w/performance metrics
	logOverallResults_trainAndValCombined(wandb_run=run, metrics_list=[performance_metrics_train, performance_metrics_test], metrics_by_cation_family_list=[performance_metrics_by_cation_family_train, performance_metrics_by_cation_family_test], metrics_by_alkyl_chain_length_list=[performance_metrics_by_alkyl_chain_length_train, performance_metrics_by_alkyl_chain_length_test])
	
	#Stop W&B run
	run.finish()

	#run feature importance analysis if linear model
	if(model_type == 'linear_l1'):
		feature_importance_analysis_trainAndValCombined(model_pipeline, train_test_sets_dict, model_basename, input_features_to_use, directory_to_save_results, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)

	return [performance_metrics_train, performance_metrics_test]


#train_dummy_regressor_no_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains dummy regressor to predict the desired property (y) (output is always the mean of the training set) 
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model
def train_dummy_regressor_no_log_y_cv(train_and_val_sets_dict):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': False,
			'ionic_radii_estimation_method': ionic_radii_estimation_method,
			'input_features': input_features_to_use
		}
	else:
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': False,
			'input_features': input_features_to_use
		}
	pipelined_model_dummy_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), DummyRegressor(strategy = 'mean'))
	model_basename_str = 'dummy_no_log_y'
	overall_performance_metrics = train_model_no_log_y_cv(train_and_val_sets_dict,pipelined_model_dummy_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics
	
#train_dummy_regressor_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains dummy regressor to predict the log(desired property) (log(y)) (output is always the mean of the training set) 
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model
def train_dummy_regressor_log_y_cv(train_and_val_sets_dict):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': True,
			'ionic_radii_estimation_method': ionic_radii_estimation_method,
			'input_features': input_features_to_use
		}
	else:
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': True,
			'input_features': input_features_to_use
		}
	pipelined_model_dummy_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), DummyRegressor(strategy = 'mean'))
	model_basename_str = 'dummy_log_y'
	overall_performance_metrics = train_model_log_y_cv(train_and_val_sets_dict,pipelined_model_dummy_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics

#train_linear_l1_model_no_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains linear model (w/L1 regularization) to predict the desired property (y), loops over different alpha (regularization strength) values
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model for each hyperparameter (nested dictionary, first set of keys = hyperparameter value)
# Approach:
# - for each value of the hyperparameter to vary (ex: alpha):
#     - for each of the k folds:
#         - train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#     - compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_linear_l1_model_no_log_y_cv(train_and_val_sets_dict):
	#Create array w/values of hyperparameter to loop over
	#Dhakal & Shah (2022) tested log(alpha) values in range [-6,50], found best was 1e-5 for lasso/L1 regularization
	alpha_values = np.power(10.0,np.arange(start=-6,stop=51)) 	
	#create dictionary to store results for each hyperparameter
	overall_performance_metrics_all_hyperparameters = {}
	#Run cross validation, evaluate different alphas
	for curr_alpha in alpha_values:
		#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
		if(ionic_radii_estimation_method != 'NA'):
			#add ionic radii estimation method in config params
			model_config_params = {
				'linear_l1_model': True,
				'trained_on_log': False,
				'alpha': curr_alpha,
				'ionic_radii_estimation_method': ionic_radii_estimation_method,
				'input_features': input_features_to_use
			}
		else:
			model_config_params = {
				'linear_l1_model': True,
				'trained_on_log': False,
				'alpha': curr_alpha,
				'input_features': input_features_to_use
			}
		pipelined_model_l1_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), Lasso(alpha = curr_alpha, max_iter=100000))
		model_basename_str = 'l1_no_log_y' + '_alpha_' + str(curr_alpha)
		overall_performance_metrics_all_hyperparameters[str(curr_alpha)] = train_model_no_log_y_cv(train_and_val_sets_dict,pipelined_model_l1_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_all_hyperparameters
	
#train_linear_l1_model_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains linear model (w/L1 regularization) to predict the log(desired property) (log(y)), loops over different alpha (regularization strength) values
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model for each hyperparameter (nested dictionary, first set of keys = hyperparameter value)
# Approach:
# - for each value of the hyperparameter to vary (ex: alpha):
#     - for each of the k folds:
#         - train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#     - compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_linear_l1_model_log_y_cv(train_and_val_sets_dict):
	#Create array w/values of hyperparameter to loop over
	#Dhakal & Shah (2022) tested log(alpha) values in range [-6,50], found best was 1e-5 for lasso/L1 regularization
	alpha_values = np.power(10.0,np.arange(start=-6,stop=51)) 	
	#create dictionary to store results for each hyperparameter
	overall_performance_metrics_all_hyperparameters = {}
	#Run cross validation, evaluate different alphas
	for curr_alpha in alpha_values:
		#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
		if(ionic_radii_estimation_method != 'NA'):
			#add ionic radii estimation method in config params
			model_config_params = {
			'linear_l1_model': True,
			'trained_on_log': True,
			'alpha': curr_alpha,
			'ionic_radii_estimation_method': ionic_radii_estimation_method,
			'input_features': input_features_to_use

			}
		else:
			model_config_params = {
				'linear_l1_model': True,
				'trained_on_log': True,
				'alpha': curr_alpha,
				'input_features': input_features_to_use
			}
		pipelined_model_l1_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), Lasso(alpha = curr_alpha, max_iter=100000))
		model_basename_str = 'l1_log_y'+ '_alpha_' +  str(curr_alpha)
		overall_performance_metrics_all_hyperparameters[str(curr_alpha)] = train_model_log_y_cv(train_and_val_sets_dict,pipelined_model_l1_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_all_hyperparameters

#train_xgb_model_no_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains XGBoost model to predict the desired property (y), loops over different max_depth values & n_estimators values
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model for each hyperparameter (nested dictionary, first set of keys = hyperparameter value)
# Approach:
# - for each value of the hyperparameter to vary (ex: alpha):
#     - for each of the k folds:
#         - train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#     - compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_xgb_model_no_log_y_cv(train_and_val_sets_dict):
	#Create array w/values of hyperparameter to loop over, based on hyperparameters tested by Dhakal & Shah (2022)
	max_depth_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
	n_estimators_values = np.arange(10,1010,10) #also vary number of estimators from [10,1000] (step size = 10)

	#create dictionary to store results for each hyperparameter
	overall_performance_metrics_all_hyperparameters = {}
	#Run cross validation, evaluate different max_depth & n_estimators
	for curr_n_estimators in n_estimators_values:
		for curr_max_depth in max_depth_values:
			#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
			if(ionic_radii_estimation_method != 'NA'):
				#add ionic radii estimation method in config params
				model_config_params = {
				'xgb_model': True,
				'trained_on_log': False,
				'n_estimators': curr_n_estimators, 
				'max_depth': curr_max_depth,
				'learning_rate': 0.07, 
				'subsample': 0.4,
				'colsample_bytree': 0.8,
				'colsample_bylevel': 0.1,
				'random_state': RNG_SEED,
				'ionic_radii_estimation_method': ionic_radii_estimation_method,
				'input_features': input_features_to_use
				}
			else:
				model_config_params = {
					'xgb_model': True,
					'trained_on_log': False,
					'n_estimators': curr_n_estimators, 
					'max_depth': curr_max_depth,
					'learning_rate': 0.07, 
					'subsample': 0.4,
					'colsample_bytree': 0.8,
					'colsample_bylevel': 0.1,
					'random_state': RNG_SEED,
					'input_features': input_features_to_use
				}
			pipelined_model_xgb_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), xgb.XGBRegressor(n_jobs=1, n_estimators = curr_n_estimators, max_depth = curr_max_depth, learning_rate=0.07, subsample=0.4, colsample_bytree=0.8, colsample_bylevel=0.1, random_state=RNG_SEED))
			model_basename_str = 'xgb_no_log_y' + '_max_depth_' + str(curr_max_depth) + '_n_estimators_' +  str(curr_n_estimators)
			overall_performance_metrics_all_hyperparameters[str(curr_max_depth)] = train_model_no_log_y_cv(train_and_val_sets_dict,pipelined_model_xgb_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_all_hyperparameters
	
#train_xgb_model_log_y_cv takes in a dictionary w/combined train_and_val sets (x,y, dataset_info)
#trains XGBoost to predict the log(desired property) (log(y)), loops over different max_depth values & n_estimators values
#computes performance metrics on validation set for each fold, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#returns dictionary w/overall performance metrics for the model for each hyperparameter (nested dictionary, first set of keys = hyperparameter value)
# Approach:
# - for each value of the hyperparameter to vary (ex: alpha):
#     - for each of the k folds:
#         - train model using the current hyperparameter value on k-1 folds, then apply model to the kth fold & compute performance metrics
#     - compute avg_performance_metrics_CV_curr_hyperparameter_value = (1/k)*sum(performance metrics when holding out each of the k folds)
def train_xgb_model_log_y_cv(train_and_val_sets_dict):
	#Create array w/values of hyperparameter to loop over, based on hyperparameters tested by Dhakal & Shah (2022)
	max_depth_values = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
	n_estimators_values = np.arange(10,1010,10) #also vary number of estimators from [10,1000] (step size = 10)

	#create dictionary to store results for each hyperparameter
	overall_performance_metrics_all_hyperparameters = {}
	#Run cross validation, evaluate different max_depth & n_estimators
	for curr_n_estimators in n_estimators_values:
		for curr_max_depth in max_depth_values:
			#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
			if(ionic_radii_estimation_method != 'NA'):
				#add ionic radii estimation method in config params
				model_config_params = {
				'xgb_model': True,
				'trained_on_log': True,
				'n_estimators': curr_n_estimators, 
				'max_depth': curr_max_depth,
				'learning_rate': 0.07, 
				'subsample': 0.4,
				'colsample_bytree': 0.8,
				'colsample_bylevel': 0.1,
				'random_state': RNG_SEED,
				'ionic_radii_estimation_method': ionic_radii_estimation_method,
				'input_features': input_features_to_use
				}
			else:
				model_config_params = {
					'xgb_model': True,
					'trained_on_log': True,
					'n_estimators': curr_n_estimators, 
					'max_depth': curr_max_depth,
					'learning_rate': 0.07, 
					'subsample': 0.4,
					'colsample_bytree': 0.8,
					'colsample_bylevel': 0.1,
					'random_state': RNG_SEED,
					'input_features': input_features_to_use
				}
			pipelined_model_xgb_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), xgb.XGBRegressor(n_jobs=1, n_estimators = curr_n_estimators, max_depth = curr_max_depth, learning_rate=0.07, subsample=0.4, colsample_bytree=0.8, colsample_bylevel=0.1, random_state=RNG_SEED))
			model_basename_str = 'xgb_log_y' + '_max_depth_' + str(curr_max_depth) + '_n_estimators_' +  str(curr_n_estimators)
			overall_performance_metrics_all_hyperparameters[str(curr_max_depth)] = train_model_log_y_cv(train_and_val_sets_dict,pipelined_model_xgb_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_all_hyperparameters


#train_dummy_regressor_on_trainAndVal_no_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info)
#trains dummy regressor to predict the desired property (y) on the combined training & validation set (output is always the mean of the training (=training + validation) set) 
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_dummy_regressor_on_trainAndVal_no_log_y(train_val_test_sets_dict):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean':  True,
			'trained_on_log': False,
			'ionic_radii_estimation_method':  ionic_radii_estimation_method,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	else:
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': False,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_dummy_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), DummyRegressor(strategy = 'mean'))
	model_basename_str = 'dummy_no_log_y_combinedTrainAndVal'
	overall_performance_metrics_list = train_model_on_trainAndVal_no_log_y(train_val_test_sets_dict,pipelined_model_dummy_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list
	
#train_dummy_regressor_on_trainAndVal_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info)
#trains dummy regressor to predict the log(desired property) (log(y)) on the combined training & validation set (output is always the mean of the training (=training + validation) set) 
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_dummy_regressor_on_trainAndVal_log_y(train_val_test_sets_dict):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': True,
			'ionic_radii_estimation_method': ionic_radii_estimation_method,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	else:
		#add ionic radii estimation method in config params
		model_config_params = {
			'dummy_regressor_mean': True,
			'trained_on_log': True,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_dummy_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), DummyRegressor(strategy = 'mean'))
	model_basename_str = 'dummy_log_y_combinedTrainAndVal'
	overall_performance_metrics_list = train_model_on_trainAndVal_log_y(train_val_test_sets_dict,pipelined_model_dummy_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list

#train_linear_l1_model_on_trainAndVal_no_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info) & optimal alpha value
#trains linear model (w/L1 regularization) to predict the desired property (y) on the combined training & validation set
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_linear_l1_model_on_trainAndVal_no_log_y(train_val_test_sets_dict, opt_alpha):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
			'linear_l1_model': True,
			'trained_on_log': False,
			'alpha': opt_alpha,
			'ionic_radii_estimation_method': ionic_radii_estimation_method,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	else:
		model_config_params = {
			'linear_l1_model': True,
			'trained_on_log': False,
			'alpha': opt_alpha,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_l1_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), Lasso(alpha = opt_alpha, max_iter=100000))
	model_basename_str = 'l1_no_log_y' + '_opt_alpha_' + str(opt_alpha) + '_combinedTrainAndVal'
	overall_performance_metrics_list= train_model_on_trainAndVal_no_log_y(train_val_test_sets_dict,pipelined_model_l1_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list
	
#train_linear_l1_model_on_trainAndVal_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info) & optimal alpha value
#trains linear model (w/L1 regularization) to predict the log(desired property) (log(y)) on the combined training & validation set
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_linear_l1_model_on_trainAndVal_log_y(train_val_test_sets_dict, opt_alpha):
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
		'linear_l1_model': True,
		'trained_on_log': True,
		'alpha': opt_alpha,
		'ionic_radii_estimation_method': ionic_radii_estimation_method,
		'input_features': input_features_to_use,
		'no_cv': True,
		'combinedTrainAndVal': True

		}
	else:
		model_config_params = {
			'linear_l1_model': True,
			'trained_on_log': True,
			'alpha': opt_alpha,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_l1_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), Lasso(alpha = opt_alpha, max_iter=100000))
	model_basename_str = 'l1_log_y'+ '_opt_alpha_' + str(opt_alpha) + '_combinedTrainAndVal'
	overall_performance_metrics_list= train_model_on_trainAndVal_log_y(train_val_test_sets_dict,pipelined_model_l1_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list

#train_xgb_model_on_trainAndVal_no_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info) & optimal max_depth & n_estimator values in a list opt_max_depth_n_estimators = [opt_max_depth, opt_n_estimators]
#trains XGBoost model to predict the desired property (y) on the combined training and validation set
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_xgb_model_on_trainAndVal_no_log_y(train_val_test_sets_dict, opt_max_depth_n_estimators):
	opt_max_depth = opt_max_depth_n_estimators[0]
	opt_n_estimators = opt_max_depth_n_estimators[1]
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
		'xgb_model': True,
		'trained_on_log': False,
		'n_estimators': opt_n_estimators, 
		'max_depth': opt_max_depth,
		'learning_rate': 0.07, 
		'subsample': 0.4,
		'colsample_bytree': 0.8,
		'colsample_bylevel': 0.1,
		'random_state': RNG_SEED,
		'ionic_radii_estimation_method': ionic_radii_estimation_method,
		'input_features': input_features_to_use,
		'no_cv': True,
		'combinedTrainAndVal': True
		}
	else:
		model_config_params = {
			'xgb_model': True,
			'trained_on_log': False,
			'n_estimators': opt_n_estimators, 
			'max_depth': opt_max_depth,
			'learning_rate': 0.07, 
			'subsample': 0.4,
			'colsample_bytree': 0.8,
			'colsample_bylevel': 0.1,
			'random_state': RNG_SEED,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_xgb_no_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), xgb.XGBRegressor(n_jobs=1, n_estimators = opt_n_estimators, max_depth = opt_max_depth, learning_rate=0.07, subsample=0.4, colsample_bytree=0.8, colsample_bylevel=0.1, random_state=RNG_SEED, importance_type='gain'))
	model_basename_str = 'xgb_no_log_y'+ '_opt_max_depth_' + str(opt_max_depth) + '_opt_n_estimators_' +  str(opt_n_estimators) + '_combinedTrainAndVal'
	overall_performance_metrics_list = train_model_on_trainAndVal_no_log_y(train_val_test_sets_dict,pipelined_model_xgb_no_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list
	
#train_xgb_model_on_trainAndVal_log_y takes in a dictionary w/training, validation, and test sets (x,y, dataset_info) & optimal max_depth & n_estimator values in a list opt_max_depth_n_estimators = [opt_max_depth, opt_n_estimators]
#trains XGBoost to predict the log(desired property) (log(y)) on the combined training and validation set
#computes performance metrics on train/test sets, and breaks down model performance by cation family & longest alkyl chain length
#logs results w/Weights & Biases (only when comparing y vs yhat, not log(y) vs log(yhat))
#code saves csv files w/results and creates plots of results (parity plot, residual plots)
#returns list of dictionaries for train/test sets w/overall performance metrics for the model
def train_xgb_model_on_trainAndVal_log_y(train_val_test_sets_dict, opt_max_depth_n_estimators):
	opt_max_depth = opt_max_depth_n_estimators[0]
	opt_n_estimators = opt_max_depth_n_estimators[1]
	#Create config dictionary for saving/logging model inputs & hyperparameters in W&B
	if(ionic_radii_estimation_method != 'NA'):
		#add ionic radii estimation method in config params
		model_config_params = {
		'xgb_model': True,
		'trained_on_log': True,
		'n_estimators': opt_n_estimators, 
		'max_depth': opt_max_depth,
		'learning_rate': 0.07, 
		'subsample': 0.4,
		'colsample_bytree': 0.8,
		'colsample_bylevel': 0.1,
		'random_state': RNG_SEED,
		'ionic_radii_estimation_method': ionic_radii_estimation_method,
		'input_features': input_features_to_use,
		'no_cv': True,
		'combinedTrainAndVal': True
		}
	else:
		model_config_params = {
			'xgb_model': True,
			'trained_on_log': True,
			'n_estimators': opt_n_estimators, 
			'max_depth': opt_max_depth,
			'learning_rate': 0.07, 
			'subsample': 0.4,
			'colsample_bytree': 0.8,
			'colsample_bylevel': 0.1,
			'random_state': RNG_SEED,
			'input_features': input_features_to_use,
			'no_cv': True,
			'combinedTrainAndVal': True
		}
	pipelined_model_xgb_log_y = pipeline.make_pipeline(sklearn.preprocessing.MinMaxScaler(), xgb.XGBRegressor(n_jobs=1, n_estimators = opt_n_estimators, max_depth = opt_max_depth, learning_rate=0.07, subsample=0.4, colsample_bytree=0.8, colsample_bylevel=0.1, random_state=RNG_SEED, importance_type='gain'))
	model_basename_str = 'xgb_log_y'+ '_opt_max_depth_' + str(opt_max_depth) + '_opt_n_estimators_' +  str(opt_n_estimators) + '_combinedTrainAndVal'
	overall_performance_metrics_list = train_model_on_trainAndVal_log_y(train_val_test_sets_dict,pipelined_model_xgb_log_y,model_config_params,model_basename_str)
	return overall_performance_metrics_list


#train_models will call relevant functions based on arguments passed into the code to either train models using 5-fold cross validation & log results, 
#or to train model with a specified optimal hyperparameter & log the results
def train_models():
	#read in dataset splits
	train_val_test_splits_dict = read_in_dataset_splits()

	#check no overlaps in train/val/test splits
	verify_no_overlaps_between_splits(train_val_test_splits_dict)

	#drop/remove any highly correlated features or features that are zero or have standard deviation=0 on training set
	train_val_test_splits_dict_removed_correlated_features = remove_highly_correlated_zero_features(train_val_test_splits_dict)

	train_and_val_sets_combined_dict = combine_train_and_val_sets(train_val_test_splits_dict_removed_correlated_features)

	if(do_cross_validation == 'True'):
		if(compare_to_dummy_regressor == 'True'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_dummy_regressor_no_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, dummy regressor trained on y: ****')
				print(overall_model_performance) 
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_dummy_regressor_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, dummy regressor trained on log(y): ****')
				print(overall_model_performance_log_y)
				print('\n')
		if(model_type == 'linear_l1'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_linear_l1_model_no_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on y: ****')
				print(overall_model_performance)
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_linear_l1_model_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on log(y): ****')
				print(overall_model_performance_log_y)
				print('\n')
		elif(model_type == 'xgboost'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_xgb_model_no_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost model trained on y: ****')
				print(overall_model_performance)
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_xgb_model_log_y_cv(train_and_val_sets_combined_dict)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost trained on log(y): ****')
				print(overall_model_performance_log_y)
				print('\n')
		else:
			sys.exit(f'Invalid model_type provided (input was : {model_type}, options are: linear_l1, xgboost)')
	else: #train on full training set, calculate metrics on validation & test sets
		if(compare_to_dummy_regressor == 'True'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_dummy_regressor_on_trainAndVal_no_log_y(train_val_test_splits_dict_removed_correlated_features)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, dummy regressor trained on y, combined train and val sets: ****')
				print(overall_model_performance)
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_dummy_regressor_on_trainAndVal_log_y(train_val_test_splits_dict_removed_correlated_features)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, dummy regressor trained on log(y), combined train and val sets: ****')
				print(overall_model_performance_log_y)
				print('\n')

		if(model_type == 'linear_l1'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_linear_l1_model_on_trainAndVal_no_log_y(train_val_test_splits_dict_removed_correlated_features, opt_hyperparameter_val)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on y, combined train and val sets: ****')
				print(overall_model_performance)
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_linear_l1_model_on_trainAndVal_log_y(train_val_test_splits_dict_removed_correlated_features, opt_hyperparameter_val)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, linear model w/L1 regularization trained on log(y), combined train and val sets: ****')
				print(overall_model_performance_log_y)
				print('\n')
		elif(model_type == 'xgboost'):
			if(molar_conductivity_or_ionicity == 'ionicity'):
				#no log transformation
				overall_model_performance = train_xgb_model_on_trainAndVal_no_log_y(train_val_test_splits_dict_removed_correlated_features, opt_hyperparameter_val)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost model trained on y, combined train and val sets: ****')
				print(overall_model_performance)
				print('\n')
			else:
				#conductivity - apply log transformation
				overall_model_performance_log_y = train_xgb_model_on_trainAndVal_log_y(train_val_test_splits_dict_removed_correlated_features, opt_hyperparameter_val)
				print(f'\n****Overall {molar_conductivity_or_ionicity} model performance, xgboost trained on log(y), combined train and val sets: ****')
				print(overall_model_performance_log_y)
				print('\n')
		else:
			sys.exit(f'Invalid model_type provided (input was : {model_type}, options are: linear_l1, xgboost)')

#Run code/train models
train_models()