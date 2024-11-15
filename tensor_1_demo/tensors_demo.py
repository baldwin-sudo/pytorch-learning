import torch
from torchviz import make_dot

# Create a tensor with requires_grad set to True
x = torch.tensor(1.0, requires_grad=True)

# Define the operations
y = 2 * x**2
z = 2 * y + x**3

# Retain gradients for y and z
# y.retain_grad()
# z.retain_grad()

# Print the tensors
print("x:\n", x)
print("y:\n", y)
print("z:\n", z)

# z.backward()
# Create the computation graph for z
graph = make_dot(z,show_attrs=True,show_saved=True, params={"x": x})

# Display the graph
graph.render("computation_graph", format="png", view=True)  # Saves as PNG and opens the image
