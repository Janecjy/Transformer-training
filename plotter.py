import re
import sys
import matplotlib.pyplot as plt

# File path where the log data is located
file_path = sys.argv[1]  # Replace this with the actual log file path

# Initialize an empty list to store the loss values
losses = []

# Read the file and extract loss values
with open(file_path, 'r') as file:
    for line in file:
        # Look for lines containing "Epoch loss" and extract the value
        match = re.search(r"Epoch loss = ([\d\.]+)", line)
        if match:
            loss_value = float(match.group(1))  # Convert the loss value to a float
            losses.append(loss_value)

# Plot the loss values over time
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o', linestyle='-', color='b')
plt.title('Loss Timeline')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.grid(True)
plt.show()
# print(file_path.split('.')[:])
plt.savefig(file_path+'.png')