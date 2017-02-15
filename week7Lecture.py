import math
import numpy as np
import matplotlib.pyplot as plt

def cost(x):
    ''' Returns the cost of producing x units '''
    result = (2000 + 10 * x + 0.2 * math.pow(x,2)) / x
    return result
    
def cost_prime(x):
    ''' Returns the rate of change when producing x units 
    
    With this equation, it is possible that result could not be a real number. 
    Therefore, we use the try/except to attempt calculation, and then catch
    the value error if it is not a real number. 
    
    https://docs.python.org/3.4/tutorial/errors.html
    '''
    try:
        result = -2000 * math.pow(x, -2) + 0.2
    except ValueError:
        result = None
    return result

maximum_production = 500

# It sometimes help to visualize what we're doing. Lets graph our cost_prime

production_counts = np.arange(0, 501, 1) # numpy range from 0 to before 501, stepping by 1
costs = [cost(x) for x in production_counts]
cost_primes = [cost_prime(x) for x in production_counts]

# Prepare a figure
plt.figure()

# Add axes limits to the figure
plt.xlim(0, maximum_production)
plt.ylim(0, 100)

# Label our axes and graph
plt.xlabel('Production Count')
plt.ylabel('Cost')
plt.title('Cost for Production Count')

# Plot the line
plt.plot(production_counts, costs, label='Cost Curve')

plt.legend()
plt.show()

# We can see that the lowest cost is about at 100, but let's verify
# Set an initial value for minimum cost that is high.
minimum_cost = 10000
minimum_count = None

# The zip function joins our counts and costs together
# https://docs.python.org/3.4/library/functions.html#zip
for count, cost in zip(production_counts, costs):
    if cost < minimum_cost:
        minimum_cost = cost
        minimum_count = count
        
# Print our results using %s to format the minimum count as a string and insert into the function
print('The lowest cost is when producing %s units' % minimum_count)

# We can also see the minimum count by graphing the cost rate of change
# Prepare a figure
plt.figure()

# Add axes limits to the figure
plt.xlim(0, maximum_production)
plt.ylim(0, 0.4)

# Label our axes and graph
plt.xlabel('Production Count')
plt.ylabel('Cost Rate of Change')
plt.title('Cost Rate of Change for Production Count')

# Plot the line
plt.plot(production_counts, cost_primes, label='Cost Prime')

plt.legend()
plt.show()

# We see that below 100 the values are not real numbers. Anything after 
# 100 is an increasing cost of change. So this validates our result above that
# producing 100 units yields the lowest cost.