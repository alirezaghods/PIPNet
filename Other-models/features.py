import numpy as np
from scipy import stats
from scipy import signal

def mean(x):
	"""
	Return the mean of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: mean of x
	"""
	return np.mean(x)

def count_above_mean(x):
	"""
	Return the number of values higher than mean

	Parameters:
		x (1darray): a time series sequence

	Returns:
		int: the number of values higher than mean
	"""
	m = mean(x)
	return np.where(x>m)[0].size

def count_below_mean(x):
	"""
	Return the number of values lower than mean

	Parameters:
		x (1darray): a time series sequence

	Returns:
		int: the number of values lower than mean
	"""
	m = mean(x)
	return np.where(x<m)[0].size

def mean_abs_diff(x):
	"""
	Return the mean over absolute differences between subsequent time series values

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the mean over absolute differences between subsequent time series values
	"""
	return np.mean(np.abs(np.diff(x)))

def sum_abs_diff(x):
	"""
	Return the sum over absolute differences between subsequent time series values

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the sum over absolute differences between subsequent time series values
	"""
	return np.sum(np.abs(np.diff(x)))

def median(x):
	"""
	Return the median of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the median of x
	"""
	return np.median(x)

def sum(x):
	"""
	Return the sum of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the sum of x
	"""
	return np.sum(x)

def abs_energy(x):
	"""
	Return the absolute energy of the time series

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the absolute energy of the time series
	"""
	return np.dot(x, x)

def std(x):
	"""
	Return the standard deviation of the time series

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the standard deviation of the time series
	"""
	return np.std(x)

def variation_coefficient(x):
	"""
	Return the variation coefficient of the time series

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the variation coefficient of the time series
	"""
	return std(x) / mean(x)

def var(x):
	"""
	Return the variance of the time series

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the variance of the time series
	"""
	return np.var(x)

def skew(x):
	"""
	Computes the skewness of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the skewness of x
	"""
	return stats.skew(x, bias=False)


def kurtosis(x):
	"""
	Computes the kurtosis (Fisher or Pearson) of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the kurtosis (Fisher or Pearson) of x
	"""
	return stats.kurtosis(x, bias=False)
		
def number_peaks(x):
	"""
	Computes the number of prominence peaks of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		int: Computes the number of prominence peaks of x
	"""
	peaks, _ = signal.find_peaks(x, prominence=1)
	return peaks.size

def max(x):
	"""
	Return the highest value of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the highest value of x
	"""
	return np.max(x)

def min(x):
	"""
	Return the lowest value of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the highest value of x
	"""
	return np.min(x)

def quantile(x, q):
	"""
	Return the q-th quantile of x

	Parameters:
		x (1darray): a time series sequence
		q (float): quantile of sequence, between 0 and 1

	Returns:
		float: the q-th quantile of x
	"""
	return np.quantile(x, q)

def cid(x):
	"""
	Computes the Complexity-Invariant Distance 
	Batista, G. E., Wang, X., & Keogh, E. J. (2011, April). 
	A complexity-invariant distance measure for time series. 
	In Proceedings of the 2011 SIAM international conference on data mining (pp. 699-710). Society for Industrial and Applied Mathematics.

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the Complexity-Invariant Distance of x
	"""
	return np.sqrt(np.sum(np.diff(x)**2))

def entropy(x):
	"""
	Computes the entropy of x

	Parameters:
		x (1darray): a time series sequence

	Returns:
		float: the entropy of x
	"""
	return stats.entropy(x)

def exteact_all_features(x):
	"""
	Computes all the features for x
	Parameters:
		x (1darray): a time series sequence

	Returns:
		2darray: extracted features from input x
	"""
	_mean = mean(x)
	cam = count_above_mean(x)
	cbm = count_below_mean(x)
	mad = mean_abs_diff(x)
	sad = sum_abs_diff(x)
	_median = median(x)
	_sum =  sum(x)
	_abs_energy = abs_energy(x)
	_std = std(x)
	variation_coeff = variation_coefficient(x)
	_var = var(x)
	_skew = skew(x)
	_kurtosis = kurtosis(x)
	num_peaks = number_peaks(x)
	_max = max(x)
	_min = min(x)
	quantile25 = quantile(x, .25)
	quantile75 = quantile(x, .75)
	_cid = cid(x)
	# ent = entropy(x)

	return np.array([_mean, cam, cbm, mad, sad, _median, _sum, _abs_energy, _std, variation_coeff,
					  _var, _skew, _kurtosis, num_peaks, _max, _min, quantile25, quantile75, _cid])