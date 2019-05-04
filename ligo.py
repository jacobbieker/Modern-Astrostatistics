import numpy as np
import matplotlib.pyplot as plt
import itertools


def reduced_mass(mass_one, mass_two):
    return (mass_one * mass_two) / (mass_one + mass_two) ** 2


def chirp_mass(m1, m2):
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def template(m1, m2):
    """

    :param m1:
    :param m2:
    :return:
    """

    c_mass = chirp_mass(m1, m2)

    B = 16.6  # In seconds to - 5/8
    t = np.linspace(0, 0.45, 10000)
    tc = 0.48

    gw_frequency = B * c_mass ** (-5 / 8) * (tc - t) ** (-3 / 8)

    t_h = np.linspace(-450, 0, 10000)

    t_merge_h = 10

    phase = 2 * np.pi * B * c_mass ** (-5 / 8) * (-3 / 8) * (t_merge_h - t_h) ** (5 / 8)
    f = B * c_mass ** (-5 / 8) * (t_merge_h - t_h) ** (-3 / 8)
    h = f ** (2 / 3) * np.cos(phase)

    return t, gw_frequency, t_h, h, c_mass


def plotting(t, gw_frequency, t_h, h, m1, m2):
    amplitude = gw_frequency ** (2 / 3)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    colors = ['k']

    im = ax1.scatter(t, gw_frequency, label='m1=%fMSun\nm2=%fMSun' % (m1, m2), c=amplitude)
    ax1.set_ylabel('$f(t)$')
    ax1.set_xlabel('$time (s)$')
    ax1.legend()

    ax2.plot(t_h, h, label='m1=%fMSun\nm2=%fMSun' % (m1, m2), color=np.random.choice(colors))
    ax2.set_ylabel('$h (t)$')
    ax2.set_xlabel('$time (s)$')
    ax2.legend(loc='upper left')
    plt.show()

    fig.colorbar(im, ax=ax1, cmap='coolwarm', fraction=0.046, pad=0.08)


def main(m1, m2):
    t, gw_frequency, t_h, h, x = template(m1, m2)
    plotting(t, gw_frequency, t_h, h, m1, m2)


def create_m_pairs(number_of_mergers):
    m1 = np.linspace(20, 50, number_of_mergers)
    m = []
    for a, b in itertools.combinations(m1, 2):
        m.append([a, b])
    return m


number_of_mergers = 10
m = create_m_pairs(number_of_mergers)
for i in range(number_of_mergers):
    main(m[i][0], m[i][1])
