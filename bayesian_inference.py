import numpy as np
import matplotlib.pyplot as plt

t_nought = 10
A = 1

L = np.linspace(0.001,40000, 1000000)
sigma_t = 1

front = 1/np.sqrt(2*np.pi*sigma_t**2)

P_L = front * np.exp(-0.5 * (((L/(sigma_t*A))**(0.25) - t_nought)**2)/sigma_t**2) * 0.25*(L/(sigma_t*A))**(-0.75)

plt.plot(L, P_L)
plt.show()
