import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('black_scholes_prices.dat')
S0 = data[:, 0]
K = data[:, 1]
call_prices = data[:, 3]
put_prices = data[:, 4]

# Plot
plt.scatter(K, call_prices, c='blue', s=1, label='Call Prices')
plt.scatter(K, put_prices, c='red', s=1, label='Put Prices')
plt.title('Black-Scholes Option Prices vs. Strike Price')
plt.xlabel('Strike Price ($)')
plt.ylabel('Option Price ($)')
plt.legend()
plt.savefig('black_scholes_prices.png')
plt.show()
