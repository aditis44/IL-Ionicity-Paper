# #### Code to plot results parity plot & compute performance metrics
# This code is used in/called in code used to build ML models for predicting ionic conductivity & ionicity of ionic liquids
# Primary Reference: Most plotting functions based on that provided in GitHub associated with ref. 1, which, in turn, were based on that in the MLFigures package (https://github.com/kaaiian/ML_figures)


# #### References (additional reference specific to parts of the code provided where they were used)
# 1. Wang, A. Y. T., Murdock, R. J., Kauwe, S. K., Oliynyk, A. O., Gurlo, A., Brgoch, J., ... & Sparks, T. D. (2020). Machine learning for materials scientists: an introductory guide toward best practices. Chemistry of Materials, 32(12), 4954-4965.
#     - GitHub Repository associated with this paper: https://github.com/anthony-wang/BestPractices
# 2. ML_figures package: https://github.com/kaaiian/ML_figures

#Imports
import sys
import numpy as np
import sklearn
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pandas as pd

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
plt.rcParams['figure.constrained_layout.use'] = True #https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html


#Code to visualize results/create parity plot 
#Based on code provided with ref [1] (https://github.com/anthony-wang/BestPryices/blob/master/notebooks/3-modeling_classic_models.ipynb)
#Takes in true y values & predictions (yhat), label for the model to use in plots, filename for saving the plot, and label for the predicted value (ex: molar conductivity [S*m^2/mol])
# print_level = 1 (default high print level) - print out additional information as code is running (0 indicates low print level)
#log_plot_labels = False (default) - if True, then x & y labels = log(y) and log(yhat), respectively
#also saves csv file w/y and yhat values 
def create_parity_plot(y, yhat, model_label, plot_filename, predicted_value_label='', print_level=1, log_plot_labels=False):
	if(print_level==1):
		print('Creating parity plot (yhat vs y)')
		print(f'Shape of y:  {np.shape(y)}')
		print(f'Shape of yhat:  {np.shape(yhat)}')
	
	xy_max = np.max([np.max(y), np.max(yhat)]) #get max xy values (max y and max yhat) to use when plotting y=yhat line (ideal)

	fig = plt.figure(figsize=(6,6))
	plt.plot(y, yhat, 'o', ms=7, mec='k', mfc='silver') #plot y vs yhat 
	plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal') #plot y=x line/ideal result is that y = yhat
	
	plt.axis('scaled')

	#add R^2 value to plot title
	metrics = compute_score(y_true=y, y_pred=yhat)

	#add labels
	if(predicted_value_label != ''):
		plt.xlabel(f'Actual {predicted_value_label}')
		plt.ylabel(f'Predicted {predicted_value_label}')
	elif(log_plot_labels==True):
		plt.xlabel(f'log(y)')
		plt.ylabel(f'log(yhat)')
	else:
		plt.xlabel(f'y')
		plt.ylabel(f'yhat')

	plot_title = f'{model_label}, ' + r'$R^2 = $' + f'{float(metrics["R2"]):0.4}'
	plt.title(plot_title)
	#plt.legend(loc='upper left')
	#plt.legend(ncols = 2, bbox_to_anchor =(0.5,-1), loc='lower center')
	fig.legend(ncols = 2, loc='outside lower center') 

	#plt.tight_layout()
	plt.savefig(plot_filename + '.png', format = 'png', bbox_inches='tight', dpi=300)
	plt.close()

	#save csv file w/y and yhat values
	y_and_yhat_values_df = pd.DataFrame()
	if(np.shape(y)[0] != np.shape(yhat)[0]):
		sys.exit(f'Error - y & yhat are not of the same shape\nShape y: {np.shape(y)}; Shape yhat: {np.shape(yhat)}')
	if(y.ndim > 1):
		if(np.shape(y)[1] == 1):
			#this often occurs where y shape = (n_samples,1) (is a column vector), if so, just take y[0]/use np.ravel
			y = np.ravel(y)
	if(yhat.ndim > 1):
		if(np.shape(yhat)[1] == 1):
			#this often occurs where yhat shape = (n_samples,1) (is a column vector), if so, just take yhat[0]/use np.ravel
			yhat = np.ravel(yhat)

	if(log_plot_labels == True):
		y_and_yhat_values_df['log(y)'] = y
		y_and_yhat_values_df['log(yhat)'] = yhat
	else:
		y_and_yhat_values_df['y'] = y
		y_and_yhat_values_df['yhat'] = yhat
	y_and_yhat_values_df.to_csv(plot_filename + '_data.csv')
	
	return fig

#Code to visualize results/create parity plot 
#Based on code provided with ref [1] (https://github.com/anthony-wang/BestPryices/blob/master/notebooks/3-modeling_classic_models.ipynb) 
#Takes in true y values & predictions (yhat), label for the model to use in plots, filename for saving the plot, and label for the predicted value (ex: molar conductivity [S*cm^2/mol])
#print_level = 1 (default high print level) - print out additional information as code is running (0 indicates low print level)
#log_plot_labels = False (default) - if True, then x & y labels = log(y) and log(yhat), respectively
def create_parity_plot_kfolds(y, yhat, model_label, plot_filename, predicted_value_label='', print_level=1, log_plot_labels=False):
	num_folds = len(y)
	if(print_level==1):
		print('Creating parity plot (yhat vs y) using k-fold cross validation results')
		print(f'Number of folds: {num_folds}')

	#First make sure y, yhat are of the same shape
	for fold in range(num_folds):
		y_curr_fold = y[fold]
		yhat_curr_fold = yhat[fold]
		y_shape = np.shape(y_curr_fold)
		yhat_shape = np.shape(yhat_curr_fold)
		print(y_shape)
		print(yhat_shape)
		if(np.size(y_shape) > np.size(yhat_shape)):
			#this often occurs if y shape = (n_samples,1) (is a column vector) and yhat shape = (n_samples,)
			#if so, just take y[0]
			if(y_shape[1] == 1):
				y[fold] = np.ravel(y_curr_fold)
			else:
				sys.exit(f'Error - y & yhat are not of the same shape\nShape y: {y_shape}; Shape yhat: {yhat_shape}')   
		elif(np.size(y_shape) < np.size(yhat_shape)):
			sys.exit(f'Error - y & yhat are not of the same shape\nShape y: {y_shape}; Shape yhat: {yhat_shape}')  
		
		elif(np.size(y_shape) == np.size(yhat_shape)):
			#both y & yhat have the same shape
			#check if y & yhat are column vectors (n_samples, 1) - if so, convert to have shape (n_samples,)
			if(np.size(y_shape) > 1):
				#convert y & yhat to have shape (n_samples, )
				y[fold] = np.ravel(y_curr_fold)
				yhat[fold] = np.ravel(yhat_curr_fold)

	fig = plt.figure(figsize=(6,6))

	xy_max = np.max([np.max(y[0]), np.max(yhat[0])])
	xy_min = np.min([np.min(y[0]), np.min(yhat[0])])
	for k in range(num_folds):	
		y_curr_fold = y[k]
		yhat_curr_fold = yhat[k]

		if(print_level==1):
			print(f'Shape of y, fold {k}: {y_curr_fold.shape}')
			print(f'Shape of yhat, fold {k}: {yhat_curr_fold.shape}')

		xy_max_curr_fold = np.max([np.max(y_curr_fold), np.max(yhat_curr_fold)]) #get max xy values (maximum over y & yhat values) to use when plotting y=yhat line (ideal)
		xy_min_curr_fold = np.min([np.min(y_curr_fold), np.min(yhat_curr_fold)]) #get min xy values (minimum over y & yhat values) to use when plotting y=yhat line (ideal)

		if(xy_max_curr_fold > xy_max):
			xy_max = xy_max_curr_fold
		if(xy_min_curr_fold < xy_min):
			xy_min = xy_min_curr_fold

		plot_label = f'Fold {k}'
		plt.plot(y_curr_fold, yhat_curr_fold, 'o', ms=7, label = plot_label) #plot y vs yhat 
		
	plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', label='ideal') #plot y=x line/ideal result is that y = yhat

	plt.xlim(xy_min,xy_max)

	plt.ylim(xy_min,xy_max)

	plt.axis('scaled')

	#add labels
	if(predicted_value_label != ''):
		plt.xlabel(f'Actual {predicted_value_label}')
		plt.ylabel(f'Predicted {predicted_value_label}')
	elif(log_plot_labels==True):
		plt.xlabel(f'log(y)')
		plt.ylabel(f'log(yhat)')
	else:
		plt.xlabel(f'y')
		plt.ylabel(f'yhat')

	plt.title(f'{model_label}, {num_folds}-fold cross validaton' )
	#plt.legend(ncols = 2, bbox_to_anchor =(0.5,-1), loc='lower center') 
	fig.legend(ncols = 2, loc='outside lower center') 
	#plt.tight_layout()

	plt.savefig(plot_filename + '.png', format = 'png', bbox_inches='tight', dpi=300)
	plt.close()
	return fig

#code to calculate performance metrics, create parity plots & create residual plots
#uses scikit-learn functions for calculating performance metrics
def compute_score(y_true, y_pred):
	return {
		"R2": f"{r2_score(y_true, y_pred):.5}",
		"MedAE": f"{median_absolute_error(y_true, y_pred):.5}",
		"MAE": f"{mean_absolute_error(y_true, y_pred):.5}",
		"MSE": f"{mean_squared_error(y_true, y_pred):.5}"
	}



