import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# before
plt.scatter(points['X'][0], points['Y'][0], s=10) # hackyyyyyyy
plt.scatter(nodes['X'], nodes['Y'], s=10)

# after
plt.figure()
plt.scatter(points['X'], points['Y'], s=10)
plt.scatter(nodes['X'], nodes['Y'], s=10)