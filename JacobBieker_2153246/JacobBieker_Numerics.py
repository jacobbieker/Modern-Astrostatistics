"""

Jacob Bieker

"""
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def box_muller(num_samples):
    z1 = random.uniform(0,1, size=num_samples)
    z2 = random.uniform(0,1,size=num_samples)

    x1 = np.cos(2*np.pi*z2) * np.sqrt((-2)*np.log(z1))
    x2 = np.sin(2*np.pi*z2) * np.sqrt((-2)*np.log(z1))
    return x1, x2


def map_to_guass(x1,x2, u, sigma):
    """
    Changes the mean and varance to those given
    :param x1:
    :param x2:
    :return:
    """

    # First change variance

    x1 = x1 * (sigma)
    x2 = x2 * (sigma)

    # then change the mean
    x1 = x1 + u
    x2 = x2 + u

    return x1, x2


x1, x2 = box_muller(1000)

x1, x2 = map_to_guass(x1, x2, 2.7, np.pi)

min_x = np.min(x1)
max_x = np.max(x2)

gaussian_dist = np.random.normal(2.7, np.pi, 10000000)
plt.hist(x1, bins=30, density=True, histtype='step', label='BoxMuller')
plt.hist(gaussian_dist, density=True, bins=1000, histtype='step', label='Gaussian')
plt.xlabel("Value")
plt.ylabel("Probability")
plt.legend(loc='best')
plt.savefig("Part_One_Gaussian.png", dpi=300)
plt.cla()

# Part 2

def random_sample(upper_func, lower_func, xmin, xmax, ymin, ymax, num_samples):
    """
    Generates random positions within the limits set by two functions

    :return:
    """

    inputs = []
    outputs = []

    while len(outputs) < num_samples: # While the number of accepted values is less than the number of required samples
        x = np.random.uniform(size=1)[0] * (xmax - xmin) + xmin # Generate random number for X
        y = np.random.uniform(size=1)[0] * (ymax - ymin) + ymin # Generate random for Y as well
        if lower_func(x) < y < upper_func(x): # The check for if lower(x) < y < upper(x), if not, its rejected, else, accepted
            inputs.append(x)
            outputs.append(y)

    return inputs, outputs


def hrd_lower(x):
    """
    The lower limit for the HRD
    :param x:
    :return:
    """
    return np.exp(-1.1*x) + 0.9*np.exp(-0.45*(x-1.1)**2)


def hrd_upper(x):
    """
    The upper limit for the HRD
    :param x:
    :return:
    """
    return np.exp(-1.*x) + np.exp(-0.5*(x-1.1)**2)

# Approx the function by finding the mind and max values for the x and y ranges

test_vals = np.linspace(0,2,1000)

ymin = np.min([hrd_upper(test_vals), hrd_lower(test_vals)])
ymax = np.max([hrd_upper(test_vals), hrd_lower(test_vals)])
xmin = 0
xmax = 2

output_x, output_y = random_sample(lower_func=hrd_lower, upper_func=hrd_upper, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, num_samples=1000)

plt.scatter(output_x, output_y, s=1, label="Sampled Points")
plt.ylabel("Stellar Luminosity (Arbitrary Units)")
plt.xlabel("Stellar Color (Arbitrary Units)")
plt.savefig("Part_2_HRD.png", dpi=300)
plt.cla()
pairs = np.asarray(list(zip(output_y, output_x)))
print(pairs.shape)

np.savetxt("Part_2_values.txt", pairs, fmt='%1.4e', delimiter=',', header='Luminosity, Color')

# Part 3

def spiral_func(theta):
    """
    Spiral Function
    :param theta: In radians, between 0 and 2pi
    :return:
    """
    return 1.686**theta


def create_spiral(theta_values, offset=0):
    """
    Part 3.2, create the spiral arms
    :param theta_values:
    :param offset: the offset, mostly for np.pi if needed for the other spiral
    :return:
    """
    x = []
    y = []

    for theta in theta_values:
        x.append(spiral_func(theta)*np.cos(theta + offset))
        y.append(spiral_func(theta)*np.sin(theta + offset))

    return x, y

# Theta values
theta_vals = np.random.uniform(0, 2*np.pi, 1000)

a1 = create_spiral(theta_vals[:500])
a2 = create_spiral(theta_vals[500:], offset=np.pi)

plt.scatter(a1[0], a1[1], s=2, label='Spiral One')
plt.scatter(a2[0], a2[1], s=2, label='Spiral Two')
plt.legend(loc='best')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Part_3_Spiral_Arms.png", dpi=300)
plt.cla()

# Part 3.3
def create_stars(theta_values, offset=0, hrd=None):
    """
    Creates stars, optionally with HRD luminosities
    :param theta_values:
    :return:
    """
    x = []
    y = []
    for theta in theta_values:
        stddev = 2 / np.sqrt(spiral_func(theta))
        mean_x, mean_y = create_spiral([theta], offset=offset)
        x1, y1 = box_muller(1)
        x.append(map_to_guass(x1, y1, mean_x, stddev)[0])
        y.append(map_to_guass(x1, y1, mean_y, stddev)[1])

    return x, y

theta_vals = np.random.uniform(0, 2*np.pi, 1000)

ax, ay = create_stars(theta_vals[:500])
bx, by = create_stars(theta_vals[500:], offset=np.pi)

plt.scatter(ax, ay, s=2, label='Spiral One')
plt.scatter(bx, by, s=2, label='Spiral Two')
plt.legend(loc='best')
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Part_3_GoldenGalaxy_Gaussian.png", dpi=300)
plt.cla()

# Now do it with the HRD ones

colors, luminosities = random_sample(lower_func=hrd_lower, upper_func=hrd_upper, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, num_samples=1000)

# Need to set the colors between 0 and 1
colors = colors #/ np.max(colors)

import matplotlib.colors

cmap = plt.get_cmap("plasma")

plt.scatter(ax, ay, s=luminosities[:500], c=cmap(colors[:500]))
plt.scatter(bx, by, s=luminosities[500:], c=cmap(colors[500:]))
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("Part_3_GoldenGalaxy_Gaussian_WithColors.png", dpi=300)
plt.cla()

# Store stellar catalog

x = ax + bx
y = ay + by
pairs = np.asarray(list(zip(x, y, output_y, output_x)))
print(pairs.shape)
np.savetxt("Part_3_values.txt", pairs, fmt='%1.4e', delimiter=',', header='X, Y, Luminosity, Color')
