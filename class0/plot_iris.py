import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

data_path = utils.get_data_set("iris_training.csv")

column_names = ['SepalLen', 'SepalWidth', 'PetalLen', 'PetalWidth', 'Specis']
df = pd.read_csv(data_path, header=0, names=column_names)
print(df.head())
print(df.describe())

iris = np.array(df)
iris_1 = df.values

print(type(iris), type(iris_1))

plt.figure(figsize=(15, 15))
plt.suptitle("Anderson's Iris Data Set\n(Blue->Setosa | Red->Versicolor | Green->Virginica)")

for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, i * 4 + j + 1)
        if i == j:
            plt.text(0.3, 0.4, column_names[i], fontsize=16)
        else:
            plt.xlabel(column_names[i])
            plt.ylabel(column_names[j])
            plt.scatter(iris[:, i], iris[:, j], c=iris[:, 4], cmap='brg')

plt.tight_layout()
plt.show()
