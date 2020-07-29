"""
Little script to draw the neutral network.

Adapted from https://gist.github.com/craffel/2d727968c3aaebd10359

"""

import matplotlib.pyplot as plt
import numpy as np

# Set the network shape
real = False
if real:
    n_c_out = 48
    n_fc1 = 13
    n_assets = 10
    n_output = 10
    n_signals = 6
    n_days = 10
    gap = 0.01
else:
    n_c_out = 12
    n_fc1 = 4
    n_assets = 3
    n_output = 3
    n_signals = 4
    n_days = 5
    gap = 0.05

# Set general figure parameters
top = 0.95
bottom = 0.05
left = 0.2
right = 0.95

# Setup the figure
fig = plt.figure(figsize=(8, 10.5))
ax = fig.gca()
ax.axis('off')

v_spacing = (top - bottom - gap * (n_assets - 1)) / (n_c_out * n_assets)
h_spacing = (right - left) / 3
max_asset_height = (n_c_out - 1) * v_spacing
diameter = v_spacing / 4.0

# Draw CNN
for a in range(n_assets):
    asset_offset = a * (max_asset_height + gap) + max_asset_height / 2.0 + bottom
    layer_top = asset_offset + v_spacing * (n_signals - 1.0) / 2.0
    for s in range(n_signals):
        day_centers = np.empty([n_days, 2])
        for d in range(n_days):
            x = v_spacing * (d + 1)
            y = layer_top - s * v_spacing
            day_centers[d] = [x, y]
            circle = plt.Circle((x, y), diameter, color='orange', ec='k', zorder=4)
            ax.add_artist(circle)

        for d in range(n_days - 1):
            line = plt.Line2D([day_centers[d, 0], day_centers[d + 1, 0]],
                              [day_centers[d, 1], day_centers[d+ 1, 1]], c='k', alpha=0.5, lw=2)
            ax.add_artist(line)

# Connect CNN



# Draw CNN output nodes
cnn_centers = np.empty([n_assets, n_c_out, 2])
for a in range(n_assets):
    asset_offset = a * (max_asset_height + gap) + max_asset_height / 2.0 + bottom

    layer_top = asset_offset + v_spacing * (n_c_out - 1.0) / 2.0
    for n in range(n_c_out):
        x = left
        y = layer_top - n * v_spacing
        cnn_centers[a, n] = [x, y]
        circle = plt.Circle((x, y), diameter, color='blue', ec='k', zorder=4)
        ax.add_artist(circle)

# Draw 1st fully connected layer
fc1_centers = np.empty([n_assets, n_fc1, 2])
for a in range(n_assets):
    asset_offset = a * (max_asset_height + gap) + max_asset_height / 2.0 + bottom

    layer_top = asset_offset + v_spacing * (n_fc1 - 1.0) / 2.0
    for n in range(n_fc1):
        x = left + h_spacing
        y = layer_top - n * v_spacing
        fc1_centers[a, n] = [x, y]
        circle = plt.Circle((x, y), diameter, color='cyan', ec='k', zorder=4)
        ax.add_artist(circle)

# Draw output layer
out_centers = np.empty([n_output, 2])
if n_assets % 2 == 0:
    layer_top = n_assets / 2 * (max_asset_height + gap) - gap / 2.0 + bottom
else:
    layer_top = (n_assets + 1) / 2 * (max_asset_height + gap) - gap - max_asset_height/2.0 + bottom
layer_top += v_spacing * (n_output - 1.0) / 2.0

for n in range(n_output):
    x = left + 2 * h_spacing
    y = layer_top - n * v_spacing
    out_centers[n] = [x, y]
    circle = plt.Circle((x, y), diameter, color='red', ec='k', zorder=4)
    ax.add_artist(circle)

# Connect the CNN to fc1
for a in range(n_assets):
    for n1 in range(n_c_out):
        for n2 in range(n_fc1):
            line = plt.Line2D([cnn_centers[a, n1, 0], fc1_centers[a, n2, 0]],
                              [cnn_centers[a, n1, 1], fc1_centers[a, n2, 1]], c='grey', alpha=0.2, lw=1)
            ax.add_artist(line)

# Connect the fc1 to output
for a in range(n_assets):
    for n1 in range(n_fc1):
        for n2 in range(n_output):
            line = plt.Line2D([fc1_centers[a, n1, 0], out_centers[n2, 0]],
                              [fc1_centers[a, n1, 1], out_centers[n2, 1]], c='grey', alpha=0.2, lw=1)
            ax.add_artist(line)

fig.savefig('nn.png')
fig.show()
