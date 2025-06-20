import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button, TextBox, CheckButtons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

class EnhancedEventNavigator:
    """
    A class for visualizing and analyzing particle physics event data with interactive clustering controls.
    
    This class provides an interactive interface to:
    - Navigate through different events in the dataset
    - Perform DBSCAN clustering on 3D point cloud data
    - Toggle visibility of noise points
    - Adjust clustering parameters
    - View cluster statistics and 2D projections
    """

    def __init__(self, fig):
        """
        Initialize the event navigator with a matplotlib figure.
        
        Parameters:
            fig (matplotlib.figure.Figure): The figure to use for plotting
        """
        self.fig = fig
        self.df = self.load_data()
        self.event_numbers = sorted(self.df["a"].unique())
        self.current_idx = 0
        self.min_event = min(self.event_numbers)
        self.max_event = max(self.event_numbers)
        
        # Clustering parameters
        self.eps = 0.3           # Maximum distance between points in a cluster
        self.min_samples = 5     # Minimum points to form a core point
        self.show_noise = True   # Whether to display noise points
        
        # Create buttons and text box
        self.create_navigation_controls()
        
        # Plot initial event
        self.plot_current_event()

    def load_data(self):
        """
        Load and preprocess the event data from file.
        
        Returns:
            pandas.DataFrame: Preprocessed data containing event information
        """
        # Load dataset from file
        # file_path = "Data file.txt"
        file_path = "Cleaned_Data.txt"
        # columns = ["a", "b", "c", "x", "y", "z", "tb", "q"]
        columns = ["a", "x", "y", "z", "tb", "q"]
        
        print("Loading data file...")
        df = pd.read_csv(file_path, delim_whitespace=True, names=columns, skiprows=1)
        
        # Convert columns to numeric type and handle any errors
        for col in ["a", "x", "y", "z", "tb", "q"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Remove any rows with missing values in essential columns
        df = df.dropna(subset=["a", "x", "y", "z"])
        
        # Convert event column to integer
        df["a"] = df["a"].astype(int)
        
        return df

    def create_navigation_controls(self):
        """
        Create all the interactive controls in the visualization.
        Controls include:
        - Next/Previous buttons for event navigation
        - Text box for jumping to specific events
        - Controls for DBSCAN parameters (eps, min_samples)
        - Checkbox to toggle noise visibility
        """
        # Create a dedicated bottom panel area
        bottom_panel_height = 0.08
        
        # Clustering parameter controls (left side, moved right for visibility)
        self.ax_min_samples = plt.axes([0.12, 0.04, 0.12, 0.03])
        self.ax_eps = plt.axes([0.12, 0.01, 0.12, 0.03])
        
        # Navigation controls (center, tighter spacing)
        button_width = 0.12
        button_height = 0.04
        button_y = 0.02
        
        self.ax_prev = plt.axes([0.45 - button_width/2, button_y, button_width, button_height])
        self.ax_next = plt.axes([0.55 - button_width/2, button_y, button_width, button_height])
        
        # Noise toggle placement (improved, not overlapping)
        self.ax_noise_toggle = plt.axes([0.50 - 0.08, 0.065, 0.16, 0.03])
        
        # Event selector (right side)
        self.ax_textbox = plt.axes([0.80, 0.02, 0.12, 0.03])

        # Create the control widgets with improved styling
        button_props = dict(color='0.85', hovercolor='0.95')
        
        self.btn_prev = Button(self.ax_prev, 'Previous', color=button_props['color'], 
                            hovercolor=button_props['hovercolor'])
        self.btn_next = Button(self.ax_next, 'Next', color=button_props['color'],
                           hovercolor=button_props['hovercolor'])
        
        self.textbox = TextBox(self.ax_textbox, 'Event #: ', initial='')
        self.eps_box = TextBox(self.ax_eps, 'eps: ', initial=str(self.eps))
        self.min_samples_box = TextBox(self.ax_min_samples, 'min_samples: ', 
                                     initial=str(self.min_samples))
        
        # Create custom checkbox with better visibility - enlarged and clearer labeling
        self.noise_toggle = CheckButtons(self.ax_noise_toggle, ['Show Noise'], [self.show_noise])
        
        # Customize checkbox appearance
        for rect in self.noise_toggle.rectangles:
            rect.set_facecolor('lightgray')
            rect.set_edgecolor('black')
            rect.set_linewidth(1)
            # Make the checkbox slightly larger
            rect.set_width(rect.get_width() * 1.2)
            rect.set_height(rect.get_height() * 1.2)
        
        # Fix label position to avoid overlap - move text further right
        for label in self.noise_toggle.labels:
            # Get current position and move text further to the right
            current_x, current_y = label.get_position()
            label.set_position((current_x + 0.15, current_y))
            # Make font slightly larger and bolder for better visibility
            label.set_fontsize(10)
            label.set_fontweight('bold')
        
        # Connect the control events to their handlers
        self.btn_next.on_clicked(self.next_event)
        self.btn_prev.on_clicked(self.prev_event)
        self.textbox.on_submit(self.search_event)
        self.eps_box.on_submit(self.update_eps)
        self.min_samples_box.on_submit(self.update_min_samples)
        self.noise_toggle.on_clicked(self.toggle_noise)

    def perform_clustering(self, event_data):
        """
        Perform DBSCAN clustering on the event data.
        
        DBSCAN is a density-based clustering algorithm that groups together points 
        that are closely packed together, while marking points in low-density regions as noise.
        
        Parameters:
            event_data (pandas.DataFrame): DataFrame containing x, y, z coordinates for one event
            
        Returns:
            numpy.ndarray: Array of cluster labels (-1 indicates noise points)
        """
        # Extract coordinates for clustering
        X = event_data[["x", "y", "z"]].values
        
        # Standardize the features (important for consistent clustering)
        X_scaled = StandardScaler().fit_transform(X)
        
        # Perform DBSCAN clustering with current parameters
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_scaled)
        
        return clustering.labels_

    def plot_current_event(self):
        """
        Plot the current event with all its visualizations.
        
        This includes:
        - 3D scatter plot of points with cluster coloring
        - Cluster statistics panel
        - 2D projection of the data
        
        The noise points can be toggled on/off based on the show_noise flag.
        
        Navigation:
        - Use Next/Previous buttons
        - Use right/left arrow keys
        - Enter event number in the text box
        """
        # Clear the figure to start fresh
        self.fig.clear()
        event_number = self.event_numbers[self.current_idx]
        
        # Create the subplot layout
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
        
        # Create the three main plotting areas
        ax1 = self.fig.add_subplot(gs[:, 0], projection='3d')  # 3D scatter plot
        ax2 = self.fig.add_subplot(gs[0, 1])                   # Cluster statistics
        ax3 = self.fig.add_subplot(gs[1, 1])                   # 2D projection (X-Y plane)
        
        # Get event data and perform clustering
        event_data = self.df[self.df["a"] == event_number]
        cluster_labels = self.perform_clustering(event_data)
        
        # Make a copy of original data and labels for statistics
        original_labels = cluster_labels.copy()
        
        # Filter data based on noise visibility setting
        if not self.show_noise:
            # Create masks for clustered and noise points
            valid_mask = cluster_labels != -1
            event_data = event_data[valid_mask]
            cluster_labels = cluster_labels[valid_mask]
        
        # Generate colors for clusters
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot clusters in 3D
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            points = event_data[mask]
            
            if label == -1:
                # Plot noise points in gray
                ax1.scatter(points["x"], points["y"], points["z"], 
                          c='gray', marker='x', label='Noise')
            else:
                # Plot cluster points with unique colors
                ax1.scatter(points["x"], points["y"], points["z"],
                          c=[color], marker='o', 
                          label=f'Cluster {label}')
        
        # Set 3D plot labels and title
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(f"3D Scatter Plot for Event {event_number}")
        ax1.legend()
        
        # Calculate and display cluster statistics (always based on original data)
        cluster_sizes = pd.Series(original_labels).value_counts()
        total_points = len(original_labels)
        
        # Build the statistics text
        stats_text = "Cluster Statistics:\n\n"
        stats_text += f"Total Points: {total_points}\n\n"
        
        for label in sorted(cluster_sizes.index):
            size = cluster_sizes[label]
            percentage = (size / total_points) * 100
            if label == -1:
                stats_text += f"Noise Points: {size} ({percentage:.1f}%)\n"
                if not self.show_noise:
                    stats_text += "  [Currently Hidden]\n"
            else:
                stats_text += f"Cluster {label}: {size} ({percentage:.1f}%)\n"
        
        # Add note about noise visibility
        if not self.show_noise and -1 in cluster_sizes.index:
            noise_count = cluster_sizes[-1]
            stats_text += f"\nNote: {noise_count} noise points are hidden"
        
        # Display statistics text
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                verticalalignment='top', fontsize=10)
        ax2.axis('off')  # Hide axes for text panel
        
        # Create 2D projection (X-Y plane)
        for label, color in zip(unique_labels, colors):
            mask = cluster_labels == label
            points = event_data[mask]
            
            if label == -1:
                # Plot noise points
                ax3.scatter(points["x"], points["y"], 
                          c='gray', marker='x', label='Noise')
            else:
                # Plot cluster points
                ax3.scatter(points["x"], points["y"],
                          c=[color], marker='o', 
                          label=f'Cluster {label}')
        
        # Set 2D plot labels and grid
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_title("X-Y Projection")
        ax3.grid(True)
        
                    # Adjust the layout
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95, 
                          wspace=0.2, hspace=0.3)
        
        # Recreate the controls (they get cleared with fig.clear())
        self.create_navigation_controls()
        
        # Add keyboard navigation hint
        plt.figtext(0.5, 0.01, 'TIP: Use ← → arrow keys to navigate between events', 
                  ha='center', fontsize=9, style='italic')
        
        # Update the display
        self.fig.canvas.draw_idle()

    # def plot_current_event(self):
    #     """
    #     Plot the current event with simple point visualization, without clustering.
    #     Shows just the raw 3D point cloud data and 2D projections.
    #     """
    #     # Clear the figure to start fresh
    #     self.fig.clear()
    #     event_number = self.event_numbers[self.current_idx]
        
    #     # Create the subplot layout
    #     gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1])
        
    #     # Create the plotting areas
    #     ax1 = self.fig.add_subplot(gs[:, 0], projection='3d')  # 3D scatter plot
    #     ax2 = self.fig.add_subplot(gs[0, 1])                   # X-Y projection
    #     ax3 = self.fig.add_subplot(gs[1, 1])                   # X-Z projection
        
    #     # Get event data
    #     event_data = self.df[self.df["a"] == event_number]
        
    #     # Plot simple 3D points (no clustering, no colors)
    #     ax1.scatter(event_data["x"], event_data["y"], event_data["z"], 
    #                 color='blue', marker='o', alpha=0.6, s=10)
        
    #     # Set 3D plot labels and title
    #     ax1.set_xlabel("X")
    #     ax1.set_ylabel("Y")
    #     ax1.set_zlabel("Z")
    #     ax1.set_title(f"3D Point Cloud for Event {event_number}")
        
    #     # Create 2D projection (X-Y plane)
    #     ax2.scatter(event_data["x"], event_data["y"], color='blue', marker='o', alpha=0.6, s=10)
    #     ax2.set_xlabel("X")
    #     ax2.set_ylabel("Y")
    #     ax2.set_title("X-Y Projection")
    #     ax2.grid(True)
        
    #     # Create 2D projection (X-Z plane)
    #     ax3.scatter(event_data["x"], event_data["z"], color='blue', marker='o', alpha=0.6, s=10)
    #     ax3.set_xlabel("X")
    #     ax3.set_ylabel("Z")
    #     ax3.set_title("X-Z Projection")
    #     ax3.grid(True)
        
    #     # Adjust the layout
    #     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95, 
    #                     wspace=0.2, hspace=0.3)
        
    #     # Recreate the navigation controls
    #     self.create_navigation_controls()
        
    #     # Update the display
    #     self.fig.canvas.draw_idle()

    def toggle_noise(self, label):
        """
        Toggle the visibility of noise points.
        
        This method is called when the noise checkbox is clicked.
        
        Parameters:
            label (str): The label of the checkbox clicked
        """
        # Toggle the show_noise flag
        self.show_noise = not self.show_noise
        
        # Redraw the plot with the new noise visibility setting
        self.plot_current_event()

    def next_event(self, event):
        """
        Move to the next event in the dataset.
        
        Parameters:
            event: Button click event (not used)
        """
        self.current_idx = (self.current_idx + 1) % len(self.event_numbers)
        self.plot_current_event()

    def prev_event(self, event):
        """
        Move to the previous event in the dataset.
        
        Parameters:
            event: Button click event (not used)
        """
        self.current_idx = (self.current_idx - 1) % len(self.event_numbers)
        self.plot_current_event()

    def search_event(self, text):
        """
        Jump to a specific event number.
        
        Parameters:
            text (str): Event number entered in the text box
        """
        # Clear any existing error messages first
        self.clear_error_messages()
        
        try:
            # Convert input to integer
            event_num = int(text)
            
            # Check if event exists in dataset
            if event_num in self.event_numbers:
                self.current_idx = self.event_numbers.index(event_num)
                self.plot_current_event()
            else:
                # Show error message if event not found
                self.show_error_message(f'Event {event_num} not found in dataset')
        except ValueError:
            # Show error for invalid input
            self.show_error_message('Please enter a valid event number')
    
    def show_error_message(self, message):
        """
        Display an error message on the figure.
        
        Parameters:
            message (str): The error message to display
        """
        self.error_text = plt.figtext(0.5, 0.95, message, 
                                    color='red', ha='center', va='center',
                                    fontsize=11, weight='bold',
                                    bbox=dict(facecolor='white', alpha=0.8, 
                                           edgecolor='red', boxstyle='round,pad=0.5'))
        self.fig.canvas.draw_idle()
    
    def clear_error_messages(self):
        """
        Clear any error messages currently displayed on the figure.
        """
        if hasattr(self, 'error_text') and self.error_text is not None:
            self.error_text.remove()
            self.error_text = None
            self.fig.canvas.draw_idle()

    def update_eps(self, text):
        """
        Update the eps parameter for DBSCAN clustering.
        
        The eps parameter defines the maximum distance between two points
        for them to be considered part of the same neighborhood.
        
        Parameters:
            text (str): New eps value entered in the text box
        """
        # Clear any existing error messages first
        self.clear_error_messages()
        
        try:
            # Convert input to float
            self.eps = float(text)
            
            # Redraw with new parameter
            self.plot_current_event()
        except ValueError:
            # Show error for invalid input
            self.show_error_message('Please enter a valid eps value (numeric)')
            
    def update_min_samples(self, text):
        """
        Update the min_samples parameter for DBSCAN clustering.
        
        The min_samples parameter defines the minimum number of points required
        to form a dense region (core point).
        
        Parameters:
            text (str): New min_samples value entered in the text box
        """
        # Clear any existing error messages first
        self.clear_error_messages()
        
        try:
            # Convert input to integer
            self.min_samples = int(text)
            
            # Redraw with new parameter
            self.plot_current_event()
        except ValueError:
            # Show error for invalid input
            self.show_error_message('Please enter a valid min_samples value (integer)')
            

def main():
    """
    Main function to run the visualization.
    
    This function creates the main figure window and
    initializes the EnhancedEventNavigator.
    """
    # Use plt.style for consistent modern appearance
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create the main figure window with custom style
    fig = plt.figure(figsize=(20, 10), facecolor='white')
    fig.canvas.manager.set_window_title('Particle Event Analyzer')
    
    # Add a startup message with progress indicator
    startup_text = fig.text(0.5, 0.5, 'Initializing Event Analyzer...', 
                         ha='center', va='center', fontsize=16,
                         bbox=dict(boxstyle="round,pad=0.5", 
                                 facecolor='white', alpha=0.8,
                                 edgecolor='lightgray'))
    plt.draw()
    
    # Create the event navigator
    navigator = EnhancedEventNavigator(fig)
    
    # Set up keyboard navigation with additional keys
    def on_key_press(event):
        if event.key in ['right', 'n']:  # Right arrow or 'n' key
            navigator.next_event(None)
        elif event.key in ['left', 'p']:  # Left arrow or 'p' key
            navigator.prev_event(None)
        elif event.key == ' ':  # Spacebar
            navigator.next_event(None)
        elif event.key == 'h':  # Toggle help message
            # This could be expanded to show a help dialog in future versions
            pass
    
    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Remove the startup message
    startup_text.remove()
    
    # Show the visualization
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()