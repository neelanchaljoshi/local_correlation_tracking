# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load flow data
vlos = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/vlos_cleaned_langfellner.npy')
lct = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pmi/supergran/vlos_test2.npy')

# Plot the comparison with vlos on x axis and lct on y
plt.figure(figsize=(8, 8))
plt.scatter(vlos, lct, s=1, alpha=0.5)
plt.xlabel('vlos (m/s)')
plt.ylabel('LCT-derived LOS Velocity (m/s)')
plt.title('Comparison of LCT-derived LOS Velocity with vlos')
plt.xlim(-500, 500)
plt.ylim(-500, 500)
plt.plot([-500, 500], [-500, 500], 'r--', label='y=x')
plt.legend()
plt.grid()
plt.show()

# %%
# Fit a line to the scatter plot
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(vlos.flatten(), lct.flatten())
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")
# Plot the scatter plot with the fitted line
plt.figure(figsize=(8, 8))
plt.scatter(vlos, lct, s=1, alpha=0.5)
plt.xlabel('vlos (m/s)')
plt.ylabel('LCT-derived LOS Velocity (m/s)')
plt.title('Comparison of LCT-derived LOS Velocity with vlos')
plt.xlim(-500, 500)
plt.ylim(-500, 500)
plt.plot([-500, 500], [-500, 500], 'r--', label='y=x')
plt.plot([-500, 500], [intercept + slope * -500, intercept + slope * 500], 'g-', label='Fitted line')
plt.legend()
plt.grid()
# %%
