import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button, TextBox
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# Load your dataset
file_path = "New Data.txt"
columns = ["event", "b", "c", "x", "y", "z", "tb", "q"]

# Read the file with space delimiter and skip the header row
print("Loading data file...")
df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=1)

# Ensure numeric data and drop incomplete rows
for col in ["event", "x", "y", "z", "tb", "q"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with NaN values in essential columns
df = df.dropna(subset=["event", "x", "y", "z"])

# Convert event to integer
df["event"] = df["event"].astype(int)

def perform_clustering(event_data, eps=0.3, min_samples=5):
    # Extract coordinates for clustering
    X = event_data[["x", "y", "z"]].values
    
    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    
    # Add cluster labels to the data
    event_data = event_data.copy()
    event_data['cluster'] = clustering.labels_
    
    return event_data

class EventNavigator:
    def __init__(self, fig):
        self.fig = fig
        self.event_numbers = sorted(df["event"].unique())
        self.current_idx = 0
        self.min_event = min(self.event_numbers)
        self.max_event = max(self.event_numbers)
        
        # Create buttons
        self.create_navigation_buttons()
        
        # Create search box
        self.create_search_box()
        
        # Plot initial event
        self.plot_current_event()

    def create_navigation_buttons(self):
        # Add navigation buttons
        self.ax_prev = plt.axes([0.35, 0.02, 0.1, 0.04])
        self.ax_next = plt.axes([0.46, 0.02, 0.1, 0.04])
        
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_prev = Button(self.ax_prev, 'Previous')
        
        self.btn_next.on_clicked(self.next_event)
        self.btn_prev.on_clicked(self.prev_event)

    def create_search_box(self):
        # Add search box with label
        self.ax_textbox = plt.axes([0.65, 0.02, 0.1, 0.04])
        self.textbox = TextBox(self.ax_textbox, 'Event #: ', initial='')
        self.textbox.on_submit(self.search_event)

    def plot_current_event(self):
        self.fig.clear()
        event_number = self.event_numbers[self.current_idx]
        
        # Create GridSpec
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
        
        # Create axes
        ax1 = self.fig.add_subplot(gs[:, 0], projection='3d')
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, 1])
        
        event_data = df[df["event"] == event_number]
        
        # Perform clustering
        clustered_data = perform_clustering(event_data)
        
        # 3D scatter plot with clusters
        scatter = ax1.scatter(
            clustered_data["x"],
            clustered_data["y"],
            clustered_data["z"],
            c=clustered_data["cluster"],
            cmap='viridis',
            marker='o'
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f"3D Scatter Plot for Event {event_number}")
        ax1.view_init(elev=20, azim=45)
        
        # Add colorbar for clusters
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # 2D plot: Y vs. X
        ax2.scatter(event_data["x"], event_data["y"], c="r", marker="x")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"2D Plot (Y vs. X) for Event {event_number}")
        ax2.grid(True)

        # 2D plot: Y and X vs. Z
        ax3.scatter(event_data["z"], event_data["y"], c="g", label="Y", marker="o")
        ax3.scatter(event_data["z"], event_data["x"], c="m", label="X", marker="x")
        ax3.set_xlabel("Z")
        ax3.set_ylabel("Value")
        ax3.set_title(f"2D Plot (Y and X vs. Z) for Event {event_number}")
        ax3.legend()
        ax3.grid(True)
        
        # Adjust layout
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.3)
        
        # Recreate buttons and search box (they get cleared with fig.clear())
        self.create_navigation_buttons()
        self.create_search_box()
        
        self.fig.canvas.draw_idle()

    def next_event(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.event_numbers)
        self.plot_current_event()

    def prev_event(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.event_numbers)
        self.plot_current_event()

    def search_event(self, text):
        try:
            event_num = int(text)
            if event_num < self.min_event or event_num > self.max_event:
                plt.figtext(0.5, 0.975, f'Event must be between {self.min_event} and {self.max_event}', 
                          color='red', ha='center', va='center')
                self.fig.canvas.draw_idle()
                return
                
            if event_num in self.event_numbers:
                self.current_idx = self.event_numbers.index(event_num)
                self.plot_current_event()
            else:
                plt.figtext(0.5, 0.95, f'Event {event_num} not found in dataset', 
                          color='red', ha='center', va='center')
                self.fig.canvas.draw_idle()
        except ValueError:
            plt.figtext(0.5, 0.95, 'Please enter a valid event number', 
                       color='red', ha='center', va='center')
            self.fig.canvas.draw_idle()

def save_all_events_to_pdf(output_filename="events.pdf"):
    # Get unique event numbers
    event_numbers = sorted(df["event"].unique())
    
    print(f"Generating PDF for {len(event_numbers)} events...")
    
    with PdfPages(output_filename) as pdf:
        for event_number in event_numbers:
            print(f"Processing event {event_number}...")
            
            # Create a new figure for each event
            fig = plt.figure(figsize=(20, 10))
            
            # Plot the event
            event_data = df[df["event"] == event_number]
            
            # Create GridSpec
            gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
            
            # Create axes and plot
            ax1 = fig.add_subplot(gs[:, 0], projection='3d')
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
            
            # Perform clustering and plotting for PDF
            clustered_data = perform_clustering(event_data)
            
            # Plot as before...
            scatter = ax1.scatter(
                clustered_data["x"],
                clustered_data["y"],
                clustered_data["z"],
                c=clustered_data["cluster"],
                cmap='viridis',
                marker='o'
            )
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Z")
            ax1.set_title(f"3D Scatter Plot for Event {event_number}")
            ax1.view_init(elev=20, azim=45)
            
            plt.colorbar(scatter, ax=ax1, label='Cluster')

            ax2.scatter(event_data["x"], event_data["y"], c="r", marker="x")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_title(f"2D Plot (Y vs. X) for Event {event_number}")
            ax2.grid(True)

            ax3.scatter(event_data["z"], event_data["y"], c="g", label="Y", marker="o")
            ax3.scatter(event_data["z"], event_data["x"], c="m", label="X", marker="x")
            ax3.set_xlabel("Z")
            ax3.set_ylabel("Value")
            ax3.set_title(f"2D Plot (Y and X vs. Z) for Event {event_number}")
            ax3.legend()
            ax3.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"PDF saved as {output_filename}")

def show_interactive_plot():
    fig = plt.figure(figsize=(20, 10))
    navigator = EventNavigator(fig)
    plt.show()

if __name__ == "__main__":
    # For interactive visualization
    show_interactive_plot()
    
    # For saving to PDF
    # save_all_events_to_pdf("events.pdf")
