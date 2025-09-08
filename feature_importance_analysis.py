#Code used to evaluate feature importance metrics for different models used to predict ionicity or molar conductivity
#Uses preprocessed dataset created using data from NIST ILThermo database (https://ilthermo.boulder.nist.gov, Kazakov et al., Dong et al. (2007)) & ILThermoPy package (https://github.com/IvanChernyshov/ILThermoPy)

#Code takes in a scikit learn model pipeline, dictionary with pandas dataframes for x_train, y_train, x_val, y_val, x_test, y_tes
#also takes in the model_basename string (used in saving results) & name of input features used for training the models
#Model types tested/evaluated: linear model (w/L1 regularization)

#Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.inspection import permutation_importance
import shap
import pandas as pd
import os
import copy
import argparse


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


#plot_data_trends takes x_train, y_train dataframes, base directory to save results to and input feature set name, whether conductivity or ionicity was predicted & the ionic radii estimation method string
#creates plots of y/target variable vs every x feature if predicting ionicity and log(y) vs x 
def plot_data_trends(x_train_dataframe, y_train_dataframe, base_directory, input_features_str, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str):
	#Create plots of y vs feature for training data
	#Code based on that in: https://dmol.pub/ml/introduction.html
	feature_names = np.asarray(x_train_dataframe.columns)
	y_col_name = list(y_train_dataframe.columns)[0]

	if(molar_conductivity_or_ionicity == 'conductivity'):
		plot_basename = base_directory + f'/training_data_molar_conductivity_vs_{input_features_str}_plot'
		log_y_plot_basename = base_directory + f'/training_data_log_molar_conductivity_vs_{input_features_str}_plot'
		y_axis_name = list(y_train_dataframe.columns)[0]
		log_y_axis_name = f'log(Molar Conductivity)'
	else: #predicted ionicity
		plot_basename = base_directory + f'/training_data_ionicity_{ionic_radii_estimation_method_str}_vs_{input_features_str}_plot'
		log_y_plot_basename = base_directory + f'/training_data_log_ionicity_{ionic_radii_estimation_method_str}_vs_{input_features_str}_plot'
		y_axis_name = 'Ionicity'
		log_y_axis_name = f'log(Ionicity)'

	#create roughly square grid of subplots
	num_cols = int(np.sqrt(len(feature_names)))
	num_rows = int(len(feature_names)/num_cols) + int(len(feature_names)%num_cols)

	#create plot of x & y values
	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharey=True, figsize=(num_rows*4, num_cols*4), dpi=300)
	axs = axs.flatten()  # so we don't have to slice by row and column
	for i, n in enumerate(feature_names):
		ax = axs[i]
		ax.scatter(
			x_train_dataframe[n], y_train_dataframe[y_col_name].to_numpy(), s=6, alpha=0.4, color=f"C{i}"
		)  # add some color
		if i % num_cols == 0:
			ax.set_ylabel(y_axis_name)
		ax.set_xlabel(n)
	# hide empty subplots
	for i in range(len(feature_names), len(axs)):
		fig.delaxes(axs[i])
	plt.savefig(f'{plot_basename}.png')
	plt.close()
	#plt.show()

	#create plot of x & log(y) values
	fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharey=True, figsize=(num_rows*4, num_cols*4), dpi=300)
	axs = axs.flatten()  # so we don't have to slice by row and column
	for i, n in enumerate(feature_names):
		ax = axs[i]
		ax.scatter(
			x_train_dataframe[n], np.log10(y_train_dataframe[y_col_name].to_numpy()), s=6, alpha=0.4, color=f"C{i}"
		)  # add some color
		if i % num_cols == 0:
			ax.set_ylabel(log_y_axis_name)
		ax.set_xlabel(n)
	# hide empty subplots
	for i in range(len(feature_names), len(axs)):
		fig.delaxes(axs[i])
	plt.savefig(f'{log_y_plot_basename}.png')
	plt.close()


#get_coefficients_linear takes in a sklearn linear estimator/model, model_basename, along with x_data_train_df, the directory to save results to, the name of the input (x) feature set, whether predicting molar conductivity or ionicity, and the ionic radii estimation method string
#creates a plot with the top 10 most important features (highest abs(coef) value) vs the model coefficients & also saves a csv file with the model coefficients for all features
def get_coefficients_linear(linear_estimator, model_basename, x_data_train_df, base_directory, input_features_str, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str):
	feature_importance_results_basename = base_directory + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + '_' + input_features_str + '_model_coefficients'
	plot_title = ''
	if('l1' in model_basename): #L1 regularization
		plot_title += 'Linear L1 model'
	else: #L2 regularization
		plot_title += 'Linear L2 model'
	if('no_log' in model_basename): #trained on y
		plot_title += f', trained to predict {molar_conductivity_or_ionicity}'
	else: #trained on log(y)
		plot_title += f', trained to predict log({molar_conductivity_or_ionicity})'

	#Code below based partially on that given following 3 links: 
	## https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html 
	## https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
	## https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html
	linear_estimator_coeffs = linear_estimator.coef_
	if(np.shape(linear_estimator_coeffs)[0] == 1):
		#need to take transpose linear_estimator_coeffs so it is (n_features, 1) rather than (1, n_features)
		linear_estimator_coeffs = np.transpose(linear_estimator_coeffs)
	model_coefficients_all = pd.DataFrame(linear_estimator_coeffs, columns = ['Coefficients'], index = list(x_data_train_df.columns))
	model_coefficients_all.to_csv(feature_importance_results_basename + '_all.csv')
	#confirm that list features returned by feature_names_in is the same as the features in x_data_train_df.columns
	if(list(linear_estimator.feature_names_in_) != list(x_data_train_df.columns)):
		sys.exit(f'List of features returned by feature_names_in_ ({list(linear_estimator.feature_names_in_)}) does not match the columns in x_data_train ({list(x_data_train_df.columns)})')
	
	#identify the top 10 features w/the highest abs(coef)
	coefficient_abs_val_all = np.abs(model_coefficients_all['Coefficients'].to_numpy())
	sorted_coef_indices = np.argsort(coefficient_abs_val_all) 
	ten_largest_abs_val_coef_indices = sorted_coef_indices[-10:]
	model_coefficients_top_ten = pd.DataFrame()
	model_coefficients_top_ten['Coefficients'] = model_coefficients_all['Coefficients'].to_numpy()[ten_largest_abs_val_coef_indices]
	model_coefficients_top_ten.index = np.asarray(model_coefficients_all.index.to_list())[ten_largest_abs_val_coef_indices]
	model_coefficients_top_ten.plot(kind='barh', figsize = (10,10))
	plt.title(plot_title)
	plt.axvline(x=0,color='k', linestyle = '-')
	plt.tick_params(axis='x')
	plt.xlabel(f'Coefficients (input feature set = {input_features_str})')
	plt.ylabel('Top 10 features with greatest abs(coef)')
	#if include sigma profiles in features, shorten tick labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
	if('sigma_profiles' in input_features_str):
		tick_locations, curr_tick_labels = plt.yticks()
		y_axis_feature_names = []
		for tick_label_text in curr_tick_labels:
			y_axis_feature_names.append(tick_label_text.get_text())
		shortened_feature_names_list = []
		for curr_feature_name in y_axis_feature_names:
			if('sigmaProbTimesArea' in curr_feature_name):
				feature_arr = curr_feature_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				shortened_feature_names_list.append(updated_feature_label)
			else:
				shortened_feature_names_list.append(curr_feature_name)
		plt.yticks(ticks = tick_locations, labels = shortened_feature_names_list)
	plt.savefig(feature_importance_results_basename + '_top_ten_plot.png')
	plt.close()
	

#get_permutation_feature_importance_trainAndValCombined obtains feature importance values using scikit-learn permutation_importance function on the training (which includes training + validation set) and test sets 
#takes in a sklearn pipeline with the final estimator/model, model_basename, along with train_test_sets_dict, the directory to save results to, the name of the input (x) feature set, whether predicting molar conductivity or ionicity, and the ionic radii estimation method string
#creates plots (bar and boxplot) with the top 10 most important features vs the feature importance values & also saves a csv file with the feature importance values for all features for train & test sets 
def get_permutation_feature_importance_trainAndValCombined(pipelined_model, model_basename, train_test_sets_dict, base_directory, input_features_str, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str):
	#from scikit-learn documentation: "Permutation feature importance is a model inspection technique that measures 
	## the contribution of each feature to a fitted model’s statistical performance on a given tabular dataset.
	## ... particularly useful for non-linear or opaque estimators, and involves randomly shuffling the values of a single feature and observing the resulting degradation of the model’s score"
	## source: https://scikit-learn.org/stable/modules/permutation_importance.html#outline-of-the-permutation-importance-algorithm
	
	feature_importance_results_basename = base_directory + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + '_' + input_features_str + '_permutation_feature_importance'

	estimator = pipelined_model[-1]
	transformer = pipelined_model[0] #ASSUMPTION: pipelined model has only 2 components: first is the transformer (ex: MinMaxScaler), second is the estimator (ex: Lasso)
	
	#Pull out x,y data
	x_train = train_test_sets_dict['x_train']	
	x_test = train_test_sets_dict['x_test']
	feature_names = np.asarray(list(x_train.columns))

	y_train_df = train_test_sets_dict['y_train']
	y_test_df = train_test_sets_dict['y_test']
	
	y_train = y_train_df.to_numpy()
	y_test = y_test_df.to_numpy()

	#Scale x data w/minmax scaler using pipeline estimator passed into the function
	x_train_scaled = transformer.transform(x_train)
	x_test_scaled = transformer.transform(x_test)

	plot_title_train = 'Permutation feature importances (training set)\n'
	plot_title_test = 'Permutation feature importances (test set)\n'
	#Assume using Linear model w/L1 regularization
	plot_title_train += 'Linear L1 model'
	plot_title_test += 'Linear L1 model'
	if('no_log' in model_basename): #trained on y
		plot_title_train += f', trained to predict {molar_conductivity_or_ionicity}'
		plot_title_test += f', trained to predict {molar_conductivity_or_ionicity}'
	else: #trained on log(y)
		plot_title_train += f', trained to predict log({molar_conductivity_or_ionicity})'
		plot_title_test += f', trained to predict log({molar_conductivity_or_ionicity})'

	
	#code below based partially on that given following 4 links: 
	## https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html 
	## https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html
	## https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
	## https://scikit-learn.org/stable/modules/permutation_importance.html

	permutation_importance_result_train = permutation_importance(estimator, X=x_train_scaled, y=y_train, n_repeats=10, random_state=RNG_SEED, scoring = 'neg_mean_absolute_error')
	permutation_importance_result_test = permutation_importance(estimator, X=x_test_scaled, y=y_test, n_repeats=10, random_state=RNG_SEED, scoring = 'neg_mean_absolute_error')

	permutation_importances_train = pd.DataFrame(index=feature_names)
	permutation_importances_train['Mean Importance (mean decrease in neg_mean_absolute_error)'] = permutation_importance_result_train.importances_mean
	permutation_importances_train['StDev Importance (standard deviation in decrease in neg_mean_absolute_error)'] = permutation_importance_result_train.importances_std
	permutation_importances_train.to_csv(feature_importance_results_basename + '_train_set.csv')

	permutation_importances_test = pd.DataFrame(index=feature_names)
	permutation_importances_test['Mean Importance (mean decrease in neg_mean_absolute_error)'] = permutation_importance_result_test.importances_mean
	permutation_importances_test['StDev Importance (standard deviation in decrease in neg_mean_absolute_error)'] = permutation_importance_result_test.importances_std
	permutation_importances_test.to_csv(feature_importance_results_basename + '_test_set.csv')

	#identify the top 10 features w/the highest importance & create bar/boxplots with them
	sorted_importances_indices_train = permutation_importance_result_train.importances_mean.argsort()
	top_ten_importances_train = pd.Series(permutation_importance_result_train.importances_mean[sorted_importances_indices_train[-10:]], index = feature_names[sorted_importances_indices_train[-10:]])
	top_ten_raw_importances_train = pd.DataFrame(permutation_importance_result_train.importances[sorted_importances_indices_train[-10:]].T, columns = feature_names[sorted_importances_indices_train[-10:]])

	fig, ax = plt.subplots()
	top_ten_importances_train.plot.barh(xerr=permutation_importance_result_train.importances_std[sorted_importances_indices_train[-10:]], ax=ax)
	plt.title(plot_title_train)
	plt.tick_params(axis='x')
	plt.xlabel('Importance (mean decrease in neg_mean_absolute_error)')
	plt.ylabel('Top 10 features with greatest importance')
	#if include sigma profiles in features, shorten tick labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
	if('sigma_profiles' in input_features_str):
		tick_locations, curr_tick_labels = plt.yticks()
		y_axis_feature_names = []
		for tick_label_text in curr_tick_labels:
			y_axis_feature_names.append(tick_label_text.get_text())
		shortened_feature_names_list = []
		for curr_feature_name in y_axis_feature_names:
			if('sigmaProbTimesArea' in curr_feature_name):
				feature_arr = curr_feature_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				shortened_feature_names_list.append(updated_feature_label)
			else:
				shortened_feature_names_list.append(curr_feature_name)
		plt.yticks(ticks = tick_locations, labels = shortened_feature_names_list)
	plt.savefig(feature_importance_results_basename + '_top_ten_barplot_train.png')
	plt.close()

	ax = top_ten_raw_importances_train.plot.box(vert=False, whis=10)
	ax.axvline(x=0,color='k', linestyle = '--')
	ax.set_title(plot_title_train)
	ax.set_xlabel('Importance (mean decrease in neg_mean_absolute_error)')
	ax.set_ylabel('Top 10 features with greatest importance')
	#if include sigma profiles in features, shorten tick labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
	if('sigma_profiles' in input_features_str):
		tick_locations, curr_tick_labels = plt.yticks()
		y_axis_feature_names = []
		for tick_label_text in curr_tick_labels:
			y_axis_feature_names.append(tick_label_text.get_text())
		shortened_feature_names_list = []
		for curr_feature_name in y_axis_feature_names:
			if('sigmaProbTimesArea' in curr_feature_name):
				feature_arr = curr_feature_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				shortened_feature_names_list.append(updated_feature_label)
			else:
				shortened_feature_names_list.append(curr_feature_name)
		plt.yticks(ticks = tick_locations, labels = shortened_feature_names_list)
	plt.savefig(feature_importance_results_basename + '_top_ten_boxplot_train.png')
	plt.close()

	sorted_importances_indices_test = permutation_importance_result_test.importances_mean.argsort()
	top_ten_importances_test = pd.Series(permutation_importance_result_test.importances_mean[sorted_importances_indices_test[-10:]], index = feature_names[sorted_importances_indices_test[-10:]])
	top_ten_raw_importances_test = pd.DataFrame(permutation_importance_result_test.importances[sorted_importances_indices_test[-10:]].T, columns = feature_names[sorted_importances_indices_test[-10:]])

	fig, ax = plt.subplots()
	top_ten_importances_test.plot.barh(xerr=permutation_importance_result_test.importances_std[sorted_importances_indices_test[-10:]], ax=ax)
	plt.title(plot_title_test)
	plt.tick_params(axis='x')
	plt.xlabel('Importance (mean decrease in neg_mean_absolute_error)')
	plt.ylabel('Top 10 features with greatest importance')
	#if include sigma profiles in features, shorten tick labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
	if('sigma_profiles' in input_features_str):
		tick_locations, curr_tick_labels = plt.yticks()
		y_axis_feature_names = []
		for tick_label_text in curr_tick_labels:
			y_axis_feature_names.append(tick_label_text.get_text())
		shortened_feature_names_list = []
		for curr_feature_name in y_axis_feature_names:
			if('sigmaProbTimesArea' in curr_feature_name):
				feature_arr = curr_feature_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				shortened_feature_names_list.append(updated_feature_label)
			else:
				shortened_feature_names_list.append(curr_feature_name)
		plt.yticks(ticks = tick_locations, labels = shortened_feature_names_list)
	plt.savefig(feature_importance_results_basename + '_top_ten_barplot_test.png')
	plt.close()

	ax = top_ten_raw_importances_test.plot.box(vert=False, whis=10)
	ax.axvline(x=0,color='k', linestyle = '--')
	ax.set_title(plot_title_test)
	ax.set_xlabel('Importance (mean decrease in neg_mean_absolute_error)')
	ax.set_ylabel('Top 10 features with greatest importance')
	#if include sigma profiles in features, shorten tick labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
	if('sigma_profiles' in input_features_str):
		tick_locations, curr_tick_labels = plt.yticks()
		y_axis_feature_names = []
		for tick_label_text in curr_tick_labels:
			y_axis_feature_names.append(tick_label_text.get_text())
		shortened_feature_names_list = []
		for curr_feature_name in y_axis_feature_names:
			if('sigmaProbTimesArea' in curr_feature_name):
				feature_arr = curr_feature_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				shortened_feature_names_list.append(updated_feature_label)
			else:
				shortened_feature_names_list.append(curr_feature_name)
		plt.yticks(ticks = tick_locations, labels = shortened_feature_names_list)
	plt.savefig(feature_importance_results_basename + '_top_ten_boxplot_test.png')
	plt.close()

#get_shap_feature_importance_trainAndValCombined obtains shapley feature importance values using the shap package (https://shap.readthedocs.io) on the training (which includes training + validation set), and test sets 
#takes in a sklearn pipeline with the final estimator/model, model_basename, along with train_test_sets_dict, the directory to save results to, the name of the input (x) feature set, whether predicting molar conductivity or ionicity, and the ionic radii estimation method string
#creates plots (beeswarm plot) with the top 10 most important features vs the feature importance values for train & test sets 
def get_shap_feature_importance_trainAndValCombined(pipelined_model, model_basename, train_test_sets_dict, base_directory, input_features_str, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str):
	feature_importance_results_basename = base_directory + '/' + model_basename + '_' + molar_conductivity_or_ionicity + '_' + ionic_radii_estimation_method_str + '_' + input_features_str + '_shap'

	estimator = pipelined_model[-1]
	transformer = pipelined_model[0] #ASSUMPTION: pipelined model has only 2 components: first is the transformer (ex: MinMaxScaler), second is the estimator (ex: Lasso)
	#set output type for transformer to be a pandas dataframe so feature names are saved
	#https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html
	transformer.set_output(transform='pandas')
	
	#Pull out x data
	x_train_df = train_test_sets_dict['x_train']	
	x_test_df = train_test_sets_dict['x_test']
	feature_names = np.asarray(list(x_train_df.columns))
	
	#Scale x data w/minmax scaler using pipeline estimator passed into the function
	x_train_scaled = transformer.transform(x_train_df)
	x_test_scaled = transformer.transform(x_test_df)

	plot_title_train = 'SHAP values (training set)\n'
	plot_title_test = 'SHAP values (test set)\n'
	#Assume using Linear model w/L1 regularization
	plot_title_train += 'Linear L1 model'
	plot_title_test += 'Linear L1 model'

	if('no_log' in model_basename): #trained on y
		plot_title_train += f', trained to predict {molar_conductivity_or_ionicity}'
		plot_title_test += f', trained to predict {molar_conductivity_or_ionicity}'
	else: #trained on log(y)
		plot_title_train += f', trained to predict log({molar_conductivity_or_ionicity})'
		plot_title_test += f', trained to predict log({molar_conductivity_or_ionicity})'

	#code below based partially on that given following 5 links: 
	## https://shap.readthedocs.io/en/stable/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
	## https://shap.readthedocs.io/en/stable/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html
	## https://github.com/shap/shap/issues/259
	## https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html 
	## https://shap.readthedocs.io/en/latest/generated/shap.plots.beeswarm.html 
	background_dist = shap.utils.sample(x_train_scaled, 100) #select 100 instances for background distribution
	explainer = shap.Explainer(estimator.predict, background_dist)
	shap_values_train = explainer(x_train_scaled)
	shap_values_test = explainer(x_test_scaled)

	#if include sigma profiles in features, shorten feature labels for them (since due to floating point precision, name of sigma profile features became longer than necessary)
	#https://github.com/shap/shap/issues/2498
	#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
	if('sigma_profiles' in input_features_str):
		original_col_names = x_train_scaled.columns.tolist()
		updated_col_names_dict = {}
		for curr_col_name in original_col_names:
			if('sigmaProbTimesArea' in curr_col_name):
				feature_arr = curr_col_name.split('_')
				sigma_val = float(feature_arr[-1])
				sigma_val_str_round_to_3_dec_places = f'{sigma_val:.3f}'
				updated_feature_label = feature_arr[0] + r' P($\sigma$)$\times$A ($Å^2$)' + r', $\sigma$ =' + sigma_val_str_round_to_3_dec_places
				updated_col_names_dict[curr_col_name] = updated_feature_label
			else:
				updated_col_names_dict[curr_col_name] = curr_col_name

		shap_values_train.feature_names = x_train_scaled.rename(columns=updated_col_names_dict).columns.tolist()
		shap_values_test.feature_names = x_test_scaled.rename(columns=updated_col_names_dict).columns.tolist()
		
	#create beeswarm plot (default max features to display is 10 top features)
	shap.plots.beeswarm(shap_values_train, show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_beeswarm_train.png')
	plt.close()

	shap.plots.beeswarm(shap_values_test, show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_beeswarm_test.png')
	plt.close()

	#Also create bar plots w/mean SHAP value and max SHAP value 
	shap.plots.bar(shap_values_train, show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_bar_mean_SHAP_train.png')
	plt.close()
	#save mean SHAP values to csv file
	#https://github.com/shap/shap/issues/632
	mean_shap_results_train_df = pd.DataFrame()
	mean_shap_results_train_df['Mean abs(SHAP)'] = np.abs(shap_values_train.values).mean(0)
	mean_shap_results_train_df['Feature'] = shap_values_train.feature_names
	mean_shap_results_train_df.to_csv(feature_importance_results_basename + '_mean_SHAP_train.csv')

	shap.plots.bar(shap_values_test, show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_bar_mean_SHAP_test.png')
	plt.close()

	#save mean SHAP values to csv file
	#https://github.com/shap/shap/issues/632
	mean_shap_results_test_df = pd.DataFrame()
	mean_shap_results_test_df['Mean abs(SHAP)'] = np.abs(shap_values_test.values).mean(0)
	mean_shap_results_test_df['Feature'] = shap_values_test.feature_names
	mean_shap_results_test_df.to_csv(feature_importance_results_basename + '_mean_SHAP_test.csv')
	
	shap.plots.bar(shap_values_train.abs.max(0), show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_bar_max_SHAP_train.png')
	plt.close()

	shap.plots.bar(shap_values_test.abs.max(0), show=False, max_display=11)
	plt.savefig(feature_importance_results_basename + '_bar_max_SHAP_test.png')
	plt.close()


#feature_importance_analysis_trainAndValCombined is the main code for running feature importance analysis
#takes in model pipeline, train/test datasets, model name, input feature string, path to save the results to, target variable being predicted, and ionic radii estimation method used
def feature_importance_analysis_trainAndValCombined(model_pipeline, train_test_sets_dict, model_basename, input_features_to_use, directory_to_save_results, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str):
	x_train_df = train_test_sets_dict['x_train']
	y_train_df = train_test_sets_dict['y_train']
	plot_data_trends(x_train_df,y_train_df,directory_to_save_results,input_features_to_use, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)

	#estimate feature importance from models
	if('dummy' in model_basename):
		return None #cannot do feature importance w/dummy regressor, where there are no trained parameters/weights
	else:
		#Linear model
		print('Obtaining coefficients for linear model ...')
		linear_estimator = model_pipeline[-1]
		get_coefficients_linear(linear_estimator, model_basename, x_train_df, directory_to_save_results, input_features_to_use, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)
		print('Finished obtaining coefficients for linear model')
	
	#estimate importance using permutation feature importance
	print('Estimating feature importance using permutation feature importance ...')
	get_permutation_feature_importance_trainAndValCombined(model_pipeline, model_basename, train_test_sets_dict, directory_to_save_results, input_features_to_use, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)
	print('Finished estimating feature importance using permutation feature importance')

	print('Estimating SHAP values ...')
	get_shap_feature_importance_trainAndValCombined(model_pipeline, model_basename, train_test_sets_dict, directory_to_save_results, input_features_to_use, molar_conductivity_or_ionicity, ionic_radii_estimation_method_str)
	print('Finished estimating SHAP values')