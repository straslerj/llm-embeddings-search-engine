import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

x_values = np.array([124e6, 774e6, 82e6, 2.7e9, 3.8e9]).reshape(-1, 1)

y_values = np.array(
    [
        0.31182425765650745,
        0.333954009192088,
        0.2902715327100286,
        0.5697966280711603,
        0.4744308906949208,
    ]
).reshape(-1, 1)

model = LinearRegression()
model.fit(x_values, y_values)

y_pred = model.predict(x_values)

r2 = r2_score(y_values, y_pred)

plt.scatter(x_values, y_values, color="blue")
plt.plot(x_values, y_pred, color="red")
plt.title(
    f"Correlation between Number of Model Parameters and Similarity Score\n$R^2$ = {r2:.2f}"
)
plt.xlabel("Number of Model Parameters")
plt.ylabel("Mean Similarity Score")
plt.savefig("r-2_score.png", dpi=300)
