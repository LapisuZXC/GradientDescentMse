import matplotlib.pyplot as plt
from GradientDescentMse import GradientDescentMse
import pandas as pd

df = pd.read_csv("example_data.csv")  # or you can change data for yours
# you'll need to adjust sample and target
X = df.drop(columns="target")
Y = df["target"]


# parameters
threshold_list = [1e-2, 1e-3, 1e-4, 1e-5]
learning_rate_list = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

models_dict = {}
loss_dict = {}
weights_dict = {}

for threshold in threshold_list:
    models_dict[threshold] = {}
    loss_dict[threshold] = {}
    weights_dict[threshold] = {}
    for learning_rate in learning_rate_list:
        model = GradientDescentMse(
            samples=X.copy(),
            targets=Y,
            learning_rate=learning_rate,
            threshold=threshold,
        )
        model.add_constant_feature()
        model.learn()

        models_dict[threshold][learning_rate] = model
        loss_dict[threshold][learning_rate] = model.iteration_loss_dict
        weights_dict[threshold][learning_rate] = model.beta

# visualisation
fig, axes = plt.subplots(len(threshold_list), len(
    learning_rate_list), figsize=(15, 10))
fig.suptitle(
    "Learning Paths for Various Thresholds and Learning Rates", fontsize=16)

for i, threshold in enumerate(threshold_list):
    for j, learning_rate in enumerate(learning_rate_list):
        ax = axes[i, j]
        iteration_loss = loss_dict[threshold][learning_rate]
        ax.plot(iteration_loss.keys(), iteration_loss.values())
        ax.set_title(f"lr={learning_rate}, th={threshold}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE Loss")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# weights output for interpretation
for threshold in threshold_list:
    for learning_rate in learning_rate_list:
        print(f"Threshold: {threshold}, Learning Rate: {learning_rate}")
        print(f"Weights (Beta): {weights_dict[threshold][learning_rate]}")
        print("-" * 50)
