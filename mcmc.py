import numpy as np
import matplotlib.pyplot as plt

# Exercise 1

position = 0
number_of_steps = int(1e3)
possible_outcomes = [-1,1]
p=.5
p_one=p
p_minus_one = 1-p

walk_point = np.zeros(number_of_steps)

for i in range(number_of_steps):
    temp = np.random.choice(possible_outcomes, p=[p_one, p_minus_one])
    walk_point[i] = walk_point[i-1] + temp

plt.plot(walk_point, '.', c='k')
plt.show()

# Exercise 2
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

header = ['omega', 'w_0', 'w_a', 'log_likelihood']
Supernova_Seed176 = pd.read_csv('Supernova_Seed176.txt', sep="\s+", header=None)
Supernova_Seed177 = pd.read_csv('Supernova_Seed177.txt', sep="\s+", header=None)

supernova_Seed176_data = pd.DataFrame(data=Supernova_Seed176)
supernova_Seed177_data = pd.DataFrame(data=Supernova_Seed177)
supernova_Seed176_data.columns=header
supernova_Seed177_data.columns=header


omega176 = supernova_Seed176_data['omega']
w_0176 = supernova_Seed176_data['w_0']
w_a176 = supernova_Seed176_data['w_a']
log_likelihood176 = supernova_Seed176_data['log_likelihood']


omega177 = supernova_Seed177_data['omega']
w_0177 = supernova_Seed177_data['w_0']
w_a177 = supernova_Seed177_data['w_a']
log_likelihood177 = supernova_Seed177_data['log_likelihood']

fig = plt.figure(figsize=[25,10])
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

ax.scatter(omega176, w_0176, w_a176, '.', s=0.01, alpha=.1)
ax2.scatter(omega177, w_0177, w_a177, '.', s=0.01, c='orange', alpha=.1)

plt.subplots_adjust(wspace=0.1)

plt.show()

fig = plt.figure(figsize=[10,10])

ax = fig.add_subplot(321)
ax.scatter(omega176, w_0176, s=0.1, alpha=.1)
ax.set_xlabel('omega')
ax.set_ylabel('w_0')
ax.axvline(0.1, c='k', linestyle='dashed')

ax = fig.add_subplot(323)
ax.scatter(omega176, w_a176, s=0.1, alpha=.1)
ax.set_xlabel('omega')
ax.set_ylabel('w_a')

ax = fig.add_subplot(325)
ax.scatter(w_0176, w_a176, s=0.1, alpha=.1)
ax.set_xlabel('w_0')
ax.set_ylabel('w_a')



ax = fig.add_subplot(322)
ax.scatter(omega177, w_0177, s=0.1, c='orange', alpha=.1)
ax.set_xlabel('omega')
ax.set_ylabel('w_0')
ax.axvline(0.1, c='k', linestyle='dashed')

ax = fig.add_subplot(324)
ax.scatter(omega177, w_a177, s=0.1, c='orange', alpha=.1)
ax.set_xlabel('omega')
ax.set_ylabel('w_a')

ax = fig.add_subplot(326)
ax.scatter(w_0177, w_a177, s=0.1, c='orange', alpha=.1)
ax.set_xlabel('w_0')
ax.set_ylabel('w_a')

maximum_likelihood = np.max(supernova_Seed176_data['log_likelihood'])
index_correct = np.where(supernova_Seed176_data['log_likelihood']==maximum_likelihood)[0]


maximum_likelihood177 = np.max(supernova_Seed177_data['log_likelihood'])
index_correct177 = np.where(supernova_Seed177_data['log_likelihood']==maximum_likelihood177)[0]

print(
    supernova_Seed176_data.loc[index_correct],
    supernova_Seed177_data.loc[index_correct177])

fig = plt.figure(figsize=[20,10])

ax = fig.add_subplot(121)
ax.plot(omega176, log_likelihood176, '.', alpha=0.1)
ax.set_xlabel('omega')
ax.set_ylabel('log likelihood')
ax = fig.add_subplot(122)
ax.plot(omega177, log_likelihood177, '.',  c='orange', alpha=0.1)
ax.set_xlabel('omega')
ax.set_ylabel('log likelihood')

mean = 0.315
sigma = 0.017
size176 = len(omega176)
size177 = len(omega177)

gauss_data176 = np.random.normal(mean, sigma, size176)
gauss_data177 = np.random.normal(mean, sigma, size177)

joint_distribution_176 = log_likelihood176 + gauss_data176
joint_distribution_177 = log_likelihood177 + gauss_data177

def gaussian(x, x_0, sigma):
    return np.exp(-((x-x_0)**2)/(2*sigma))

dokimi = [gaussian(i, mean, sigma) for i in omega176]

fig = plt.figure(figsize=[20,10])

ax = fig.add_subplot(121)
bins = plt.hist(omega176, 500, label='SN data')
bins = plt.hist(gauss_data176, 500, color='red', label='CMB data')
bins = plt.hist(joint_distribution_176, 500, color='orange', label='Joint distribution')
ax.set_xlabel('omega')

ax.legend(loc='upper left')


ax = fig.add_subplot(122)
bins = plt.hist(omega177, 500, label='SN data')
bins = plt.hist(gauss_data177, 500, color='red', label='CMB data')
bins = plt.hist(joint_distribution_177, 500, color='orange', label='Joint distribution')
ax.set_xlabel('omega')
ax.legend(loc='upper left')

kwstas = ['weight'] + ['%i'%i for i in range(1,9)] + ['log likelihood'] + ['11']
kwstas

header = ['weight'] + ['%i'%i for i in range(1,9)] + ['log likelihood'] + ['11']
bad_mcmc = pd.read_csv('Bad_NeutrinoViscosity_Chain.txt', sep="\s+", header=None)

df = pd.DataFrame(data=bad_mcmc)
df.columns=header
log_likelihood = df['log likelihood']

fig = plt.figure(figsize=[20,10])

ax = fig.add_subplot(121)
bins = plt.plot(log_likelihood, c='k', label='log_likelihood')
ax.set_xlabel('index')
ax.set_ylabel('log_likelihood')
ax.legend(loc='upper left')


ax = fig.add_subplot(122)
bins = plt.hist(log_likelihood, 500, color='k', label='log_likelihood')
ax.set_xlabel('log likelihood')
ax.legend(loc='upper left')

from scipy.special import gamma, factorial

def d_dimensioned_sphere(d):
    return (np.pi**(d/2))/gamma(1 + d/2)

max_dimension = 20
cube_volume = np.ones(max_dimension)
sphere_volume = []
for i in range(max_dimension):
    sphere_volume.append(d_dimensioned_sphere(i))

plt.plot(sphere_volume)
plt.show()

relative_volume_sphere_over_cube = sphere_volume/cube_volume
plt.plot(relative_volume_sphere_over_cube)
plt.show()

