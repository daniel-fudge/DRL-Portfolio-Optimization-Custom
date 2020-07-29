"""
Little script to draw the neutral network.

Adapted from https://gist.github.com/craffel/2d727968c3aaebd10359

"""

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

# Select the actor or critic network
actor = False

# Set general figure parameters
top = 0.99
if actor:
    bottom = 0.05
else:
    bottom = 0.14
left = 0.2
right = 0.99
h_spacing = (right - left) / 3

# Set the network shape
real = True
kernel = 3
if real:
    cnn_alpha = 0.2
    fc1_alpha = 0.5
    n_signals = 6
    n_days = 10
    n_c_out = (n_days - kernel + 1) * n_signals
    n_fc1 = 16
    n_assets = 10
    gap = 0.005

    # Node radii
    r_input = 0.004
    r_c_out = (top - bottom - (n_assets - 1) * gap) / (n_c_out * n_assets * 3.0)
    r_fc1 = r_c_out
    r_output = 0.005
else:
    cnn_alpha = 0.8
    fc1_alpha = 0.8
    n_days = 5
    n_signals = 3
    n_c_out = (n_days - kernel + 1) * n_signals
    n_fc1 = 6
    n_assets = 4
    gap = 0.04

    # Node radii
    r_input = r_c_out = r_fc1 = r_output = 0.005

c_out_height = n_c_out * 3 * r_c_out - r_c_out
dy_asset = [a * (c_out_height + gap) + c_out_height / 2.0 + bottom for a in range(n_assets)]

# Setup the figure
fig = plt.figure(figsize=(20, 20), dpi=200)
ax = fig.gca()
ax.axis('off')

# Draw input
day_centers = np.empty([n_assets, n_signals, n_days, 2])
v_spacing = 3 * r_input
for a in range(n_assets):
    layer_top = dy_asset[a] + v_spacing * (n_signals - 1.0) / 2.0
    for s in range(n_signals):

        # Draw nodes
        for d in range(n_days):
            x = v_spacing * (d + 1)
            y = layer_top - s * v_spacing
            day_centers[a, s, d] = [x, y]
            circle = plt.Circle((x, y), radius=r_input, color='orange', ec='k', zorder=4)
            ax.add_artist(circle)

        # Draw lines
        line = plt.Line2D([day_centers[a, s, 0, 0], day_centers[a, s, -1, 0]],
                          [day_centers[a, s, 0, 1], day_centers[a, s, -1, 1]], c='k', alpha=0.5, lw=2)
        ax.add_artist(line)

# Draw CNN output nodes
cnn_centers = np.empty([n_assets, n_c_out, 2])
v_spacing = 3 * r_c_out
for a in range(n_assets):
    layer_top = dy_asset[a] + v_spacing * (n_c_out - 1.0) / 2.0
    for n in range(n_c_out):
        x = left
        y = layer_top - n * v_spacing
        cnn_centers[a, n] = [x, y]
        circle = plt.Circle((x, y), radius=r_c_out, color='blue', ec='k', zorder=4)
        ax.add_artist(circle)

# Draw connection between input and CNN
patches = []
for a in range(n_assets):
    points = np.vstack([day_centers[a, 0, -1, :], day_centers[a, -1, -1, :],
                        cnn_centers[a, -1, :], cnn_centers[a, 0, :]])
    patches.append(Polygon(points))
ax.add_collection(PatchCollection(patches, alpha=0.2, fc='gold', ec='goldenrod'))

# Draw 1st fully connected layer
fc1_centers = np.empty([n_assets, n_fc1, 2])
v_spacing = 3 * r_fc1
for a in range(n_assets):
    layer_top = dy_asset[a] + v_spacing * (n_fc1 - 1.0) / 2.0
    for n in range(n_fc1):
        x = left + h_spacing
        y = layer_top - n * v_spacing
        fc1_centers[a, n] = [x, y]
        circle = plt.Circle((x, y), radius=r_fc1, color='cyan', ec='k', zorder=4)
        ax.add_artist(circle)

# Draw action layer
action_centers = np.empty([n_assets, 2])
if not actor:
    for n in range(n_assets):
        x = left + h_spacing
        y = (n + 1) * 3 * r_output
        action_centers[n] = [x, y]
        circle = plt.Circle((x, y), radius=r_output, color='red', ec='k', zorder=4)
        ax.add_artist(circle)

# Draw output layer
if actor:
    out_centers = np.empty([n_assets, 2])
    for n in range(n_assets):
        x = left + 2 * h_spacing
        y = dy_asset[n]
        out_centers[n] = [x, y]
        circle = plt.Circle((x, y), radius=r_output, color='red', ec='k', zorder=4)
        ax.add_artist(circle)
else:
    out_centers = np.array([[left + 2 * h_spacing, 0.5]])
    circle = plt.Circle(out_centers[0], radius=r_output, color='deeppink', ec='k', zorder=4)
    ax.add_artist(circle)

# Connect the CNN to fc1
for a in range(n_assets):
    for n1 in range(n_c_out):
        for n2 in range(n_fc1):
            line = plt.Line2D([cnn_centers[a, n1, 0], fc1_centers[a, n2, 0]],
                              [cnn_centers[a, n1, 1], fc1_centers[a, n2, 1]], c='grey', alpha=cnn_alpha, lw=1)
            ax.add_artist(line)

# Connect the fc1 and actions if critic to output
if actor:
    for a in range(n_assets):
        for n1 in range(n_fc1):
            for n2 in range(n_assets):
                line = plt.Line2D([fc1_centers[a, n1, 0], out_centers[n2, 0]],
                                  [fc1_centers[a, n1, 1], out_centers[n2, 1]], c='grey', alpha=fc1_alpha, lw=1)
                ax.add_artist(line)
else:
    for a in range(n_assets):
        for n1 in range(n_fc1):
            line = plt.Line2D([fc1_centers[a, n1, 0], out_centers[0, 0]],
                              [fc1_centers[a, n1, 1], out_centers[0, 1]], c='grey', alpha=0.8, lw=1)
            ax.add_artist(line)

    for a in range(n_assets):
        line = plt.Line2D([action_centers[a, 0], out_centers[0, 0]],
                          [action_centers[a, 1], out_centers[0, 1]], c='grey', alpha=0.8, lw=1)
        ax.add_artist(line)


plt.tight_layout()
if actor:
    if real:
        name = 'network-actor-real.png'
    else:
        name = 'network-actor-simple.png'
else:
    if real:
        name = 'network-critic-real.png'
    else:
        name = 'network-critic-simple.png'

fig.savefig(name, dpi=200)
fig.show()
