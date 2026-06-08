import matplotlib.pyplot as plt

models = [
    "ARIMA",
    "LSTM",
    "GCN",
    "GAT",
    "GCN_LSTM"
]

rmse = [
    4.81,
    30.28,
    1.79,
    0.80,
    0.04
]

plt.figure(figsize=(8,5))

plt.bar(
    models,
    rmse
)

plt.title(
    "RMSE Comparison"
)

plt.xlabel(
    "Models"
)

plt.ylabel(
    "RMSE"
)

plt.savefig(
    "results/rmse_comparison.png"
)

plt.show()

print(
    "RMSE Graph Saved!"
)