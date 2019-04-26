import numpy as np

# Part 1-2

number_of_samples = np.int(1e6)
gauss_data = np.random.normal(0,1,number_of_samples)
samples_greater_than_three = 0
for i in gauss_data:
    if i>3:
        samples_greater_than_three+=1
fraction_of_samples = samples_greater_than_three/number_of_samples

print('\nPart 1,2 x~G(0,1)\nnumber of x>3 samples: %i\nfraction of samples: %f\nhypothesis rejected %i times' \
      %(samples_greater_than_three,fraction_of_samples, samples_greater_than_three))

# Part 3

# (1)
number_of_samples = np.int(1e6)
gauss_data = np.random.normal(0,np.sqrt(2),number_of_samples)
samples_greater_than_three = 0
for i in gauss_data:
    if i>3:
        samples_greater_than_three+=1
fraction_of_samples = samples_greater_than_three/number_of_samples

print('\nPart 3.1 x~G(0,2)\nnumber of x>3 samples: %i\nfraction of samples: %f\nhypothesis rejected %i times' \
      %(samples_greater_than_three,fraction_of_samples, samples_greater_than_three))

# (2)
gauss_data_m_equals_two = np.random.normal(-0.5,1,number_of_samples)
samples_greater_than_three = 0
for i in gauss_data_m_equals_two:
    if i>3:
        samples_greater_than_three+=1
fraction_of_samples = samples_greater_than_three/number_of_samples

print('\nPart 3.2 x~G(-0.5,1)\nnumber of x>3 samples: %i\nfraction of samples: %f\nhypothesis rejected %i times' \
      %(samples_greater_than_three,fraction_of_samples, samples_greater_than_three))

