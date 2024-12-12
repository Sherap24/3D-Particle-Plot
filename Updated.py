import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

# Load your dataset
# Specify the file path and column names for the dataset. The dataset is assumed to be tab-delimited.
file_path = "Data file.txt"
columns = ["event", "x", "y", "z", "tb", "q"]
df = pd.read_csv(file_path, delimiter="\t", names=columns, header=None)

# Convert columns to numeric types and remove rows with missing data
# This ensures the data is clean and usable for plotting.
df["event"] = pd.to_numeric(df["event"], errors="coerce")
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["z"] = pd.to_numeric(df["z"], errors="coerce")
df = df.dropna(subset=["event", "x", "y", "z"])  # Remove rows with NaN values in these columns
df["event"] = df["event"].astype(int)  # Convert event column to integer for indexing

# Function to plot data for a specific event
def plot_event(event_number, ax1, ax2, ax3):
    """

    Plots the data corresponding to a specific event number in three different visualizations:
    1. 3D scatter plot of (x, y, z)
    2. 2D scatter plot of y vs. x
    3. 2D scatter plots of y and x vs. z
    
    Parameters:
    - event_number: The specific event number to plot.
    - ax1, ax2, ax3: Axes objects for plotting.

    """
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Filter the dataset for the selected event
    event_data = df[df["event"] == event_number]
    if event_data.empty:
        print(f"No data available for event {event_number}")
        return

    x = event_data["x"]
    y = event_data["y"]
    z = event_data["z"]

    # Plot 3D scatter plot
    ax1.scatter(x, y, z, c="b", marker="o")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"3D Scatter Plot for Event {event_number}")

    # Plot 2D scatter plot: Y vs. X
    ax2.scatter(x, y, c="r", marker="x")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"2D Plot (Y vs. X) for Event {event_number}")

    # Plot 2D scatter plots: Y and X vs. Z
    ax3.scatter(z, y, c="g", label="Y", marker="o")
    ax3.scatter(z, x, c="m", label="X", marker="x")
    ax3.set_xlabel("Z")
    ax3.set_ylabel("Value")
    ax3.set_title(f"2D Plot (Y and X vs. Z) for Event {event_number}")
    ax3.legend()

    plt.draw()

# Interactive navigation for events
class EventNavigator:
    """

    Enable navigation through events using 'Next' and 'Previous' buttons.

    """
    def __init__(self, ax1, ax2, ax3, max_event):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.current_event = 0  # Start with the first event
        self.max_event = max_event  # Maximum event number
        plot_event(self.current_event, self.ax1, self.ax2, self.ax3)  # Initial plot

    def next_event(self, event):
        """

        Display the next event's plots. Wraps around if it reaches the end.

        """
        self.current_event = (self.current_event + 2) % (self.max_event + 2)
        plot_event(self.current_event, self.ax1, self.ax2, self.ax3)

    def prev_event(self, event):
        """

        Display the previous event's plots. Wraps around if it goes below zero.
        
        """
        self.current_event = (self.current_event - 2) % (self.max_event + 2)
        plot_event(self.current_event, self.ax1, self.ax2, self.ax3)

# Setup the figure and subplots
fig = plt.figure(figsize=(18, 10))
ax1 = fig.add_subplot(121, projection="3d")  # 3D plot
ax2 = fig.add_subplot(222)  # 2D plot: Y vs. X
ax3 = fig.add_subplot(224)  # 2D plot: Y and X vs. Z

# Initialize the event navigator
max_event_number = df["event"].max()  # Maximum event number in the dataset
navigator = EventNavigator(ax1, ax2, ax3, max_event_number)

# Add navigation buttons
axprev = plt.axes([0.4, 0.01, 0.1, 0.05])  # Position for 'Previous' button
axnext = plt.axes([0.5, 0.01, 0.1, 0.05])  # Position for 'Next' button
bnext = Button(axnext, "Next")
bnext.on_clicked(navigator.next_event)  # Bind button to 'next_event' function
bprev = Button(axprev, "Previous")
bprev.on_clicked(navigator.prev_event)  # Bind button to 'prev_event' function

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
