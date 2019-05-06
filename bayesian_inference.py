import numpy as np
import matplotlib.pyplot as plt


print((-2000**(2/3)))
print((abs(-2000)**(2/3)))
print((abs(-2000**(2/3))))
print(((-2000)**(2/3)).real)
exit()

t_nought = 10
A = 1

L = np.linspace(0.001,40000, 1000000)
sigma_t = 1

front = 1/np.sqrt(2*np.pi*sigma_t**2)


def P_L(L):
    return front * np.exp(-0.5 * (((L/(sigma_t*A))**(0.25) - t_nought)**2)/sigma_t**2) * 0.25*(L/(sigma_t*A))**(-0.75)

plt.plot(L, P_L(L))
plt.show()

def random_sample(func, xmin, xmax, ymin, ymax, num_samples):
    """
    Generates random positions that follow the profile of equation 2

    :return:
    """

    inputs = []
    outputs = []

    while len(outputs) < num_samples: # While the number of accepted values is less than the number of required samples
        x = np.random.uniform(size=1)[0] * (xmax - xmin) + xmin # Generate random number for X
        y = np.random.uniform(size=1)[0] * (ymax - ymin) + ymin # Generate random for Y as well
        if y <= func(x): # The check for if y <= p(x), if not, its rejected, else, accepted
            inputs.append(x)
            outputs.append(y)

    return inputs, outputs


x, y = random_sample(P_L, L[0], L[-1], min(P_L(L)), max(P_L(L)), num_samples=1000000)

plt.hist(x, bins=10000, density=True)
plt.title("X")
plt.show()

alpha = np.linspace(0, 20, 1000)
Posterior_alpha = pow(2,-alpha)*(alpha - 1)
plt.plot(alpha, Posterior_alpha)
plt.xlim(0,10)
plt.ylabel( r'P($\alpha$|S)', size=20)
plt.xlabel(r'$\alpha$', size=20)
plt.show()

print(alpha[np.argmax(Posterior_alpha)])

