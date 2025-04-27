import matplotlib.pyplot as plt

# Sample data (replace with actual numbers)
epochs = list(range(1, 11))
map_scores = [0.37, 0.44, 0.46, 0.48, 0.49, 0.51, 0.51, 0.52, 0.52, 0.53]
nds_scores = [0.44, 0.52, 0.55, 0.57, 0.58, 0.60, 0.59, 0.61, 0.61, 0.62]

# Plotting both in one figure for space-efficiency (good for reports)
plt.figure(figsize=(8, 5))
plt.plot(epochs, map_scores, marker='o', label='mAP')
plt.plot(epochs, nds_scores, marker='s', label='NDS')
plt.title("CenterPoint Performance vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("centerpoint_metrics_vs_epoch.png", dpi=300)
plt.show()