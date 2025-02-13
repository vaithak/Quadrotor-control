import matplotlib.pyplot as plt
import numpy as np

# Data
resolutions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dijkstra_nodes = [18902, 5946, 2334, 1049, 741, 481, 327, 196]
astar_nodes = [3038, 1036, 376, 309, 121, 72, 249, 66]

# Create figure and axis
plt.figure(figsize=(12, 6))

# Plot both lines
plt.plot(resolutions, dijkstra_nodes, 'b-o', linewidth=2, label='Dijkstra')
plt.plot(resolutions, astar_nodes, 'r-o', linewidth=2, label='A*')

# Customize the plot
plt.title('Comparison of Nodes Expanded: Dijkstra vs A*', fontsize=14, pad=20)
plt.xlabel('Resolution', fontsize=12)
plt.ylabel('Number of Nodes Expanded', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Set y-axis to logarithmic scale since there's a large range of values
#plt.yscale('log')

# Add horizontal gridlines
plt.grid(True, which="both", ls="-", alpha=0.2)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()
