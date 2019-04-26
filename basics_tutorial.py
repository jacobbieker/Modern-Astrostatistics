import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator

# Read in the covariance matrix as a vector
list = open("AllExercises/DataFiles/SN_covmat.txt").read().splitlines()
float_list = np.asfarray(list)
# Reshaping the covariance vector to sqrt(961)xsqrt(961) matrix
cov = np.reshape(float_list, (int(np.round(np.sqrt(float_list.shape[0]), 0)), int(np.round(np.sqrt(float_list.shape[0]),0))))
# Printing the length of covariance vector and the shape of covariance matrix
print('\033[1m' + 'covariance vector length:', np.shape(float_list), '\ncovariance matrix shape:',np.shape(cov))

# Calculating the correlation matrix
pddata = pd.DataFrame(data=cov)
corr = pddata.corr()
# Calculating the inverse/precision matrix
inverse = np.linalg.inv(cov)

# Plotting the Covariance, Correlation, Precision matrices
fig,(ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(25,25))

ax1 = plt.subplot(1,3,1)
im1 = ax1.imshow(cov, vmin=np.max(cov), vmax=np.min(cov), cmap='CMRmap')
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_title('Covariance matrix plot')

ax2 = plt.subplot(1,3,2)
im2 = ax2.imshow(corr, vmin=corr.values.min(), vmax=corr.values.max(), cmap='CMRmap')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax2.set_title('Correlation matrix plot')

ax3 = plt.subplot(1,3,3)
im3 = ax3.imshow(inverse, vmin=np.min(inverse), vmax=np.max(inverse), cmap='CMRmap')
cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
ax3.set_title('Precision matrix plot')


plt.show()

# Reading the values of the correlation matrix
corr_values = corr.values
# Sorting the values by their absolute value
sorted_flattened_corr_values = np.sort(np.abs(corr_values.flatten()))
# Keeping the unique values as we have double the values due to the matrix being diagonal
unique_sorted_flattened_corr_values = np.unique(sorted_flattened_corr_values)

# Printing the most and least correlated data points
print ('\033[1m' + 'The two least correlated data are:','\n\nCoordinates:', np.where(corr_values==unique_sorted_flattened_corr_values[0]),'\nValue =', unique_sorted_flattened_corr_values[0],
       '\nCoordinates: ' ,np.where(corr_values==unique_sorted_flattened_corr_values[1]), '\nValue =:',unique_sorted_flattened_corr_values[1],
       '\n\n\nWhile the most correlated data are:','\n\nCoordinates: ', np.where(corr_values==-unique_sorted_flattened_corr_values[-2]),'\nValue =', unique_sorted_flattened_corr_values[-2],
       '\nCoordinates: ', np.where(corr_values==unique_sorted_flattened_corr_values[-3]),'\nValue =', unique_sorted_flattened_corr_values[-3])

# Printing the most correlated data that contain the least information
print('\033[1m' + 'The data points that are mostly correlated, are:\n\n', unique_sorted_flattened_corr_values[-12:-2],
      '\n\nHence containing the least information and the first ones to be thrown out if necessary.')

std=[]
# Append diagonal of the covariance matrix in an array as these values contain the error
for i in pddata:
    std=np.append(std, pddata[i][i])
# Calculating standard error
err=std/np.sqrt(31)
# Appending the errors in a dictionary so I can identify which error corresponds to which row
errdic = dict(enumerate(err))
# The highest value is the one with the biggest error bar
max_err = max(errdic.items(), key=operator.itemgetter(1))

print('\033[1m' + 'The largest error bar is', max_err[1], 'and it is in the second row.')

# Calculating the determinant of the covariance matrix
det = np.linalg.det(cov)

print('\033[1m' + 'det[cov] =', det, 'which is almost but not equal to zero, therefore the inverse exists.',
      '\nA covariance matrix with exactly zero determinant would mean that all the random variables are perfectly correlated')

