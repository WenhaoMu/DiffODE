import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

with open("dataset/ellipse_branin_10000.p", "rb") as f:
    data_tuple = pickle.load(f)


coordinates = data_tuple[0] 
function_values = data_tuple[1] 
print(type(coordinates),type(coordinates[0]))

x_coords = coordinates[:, 0]
y_coords = coordinates[:, 1]
print(coordinates.shape)

plt.figure(figsize=(10, 6))
sc = plt.scatter(x_coords, y_coords, c=function_values, marker='o', cmap='viridis')
plt.colorbar(sc, label='Function Value')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Data Points with Function Values')
plt.grid(True)

optimal_points = np.array([[-3.14, 12.275], [3.14, 2.275], [9.42, 2.745]])
for point in optimal_points:
    plt.scatter(point[0], point[1], color='red')

plt.xlim([-5, 10])
plt.ylim([0, 15])

os.makedirs("./plot",exist_ok=True)
plt.savefig("./plot/constraint.jpg")
