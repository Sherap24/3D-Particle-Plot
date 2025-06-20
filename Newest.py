import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class SeagullEventCleaner:
    def __init__(self, input_file="Data file.txt", output_file="Cleaned_Data.txt", 
                 max_event=1000, eps=0.3, min_samples=5):
        """
        Initialize the seagull-focused event cleaner.
        
        This cleaner is designed to:
        1. Detect seagull-like patterns in particle physics events
        2. Remove background noise while preserving main clusters
        3. Prepare clean data for training a seagull vs non-seagull classifier
        
        Parameters:
            input_file (str): Path to the input data file
            output_file (str): Path to save the cleaned data
            max_event (int): Maximum event number to process (up to 1000)
            eps (float): DBSCAN epsilon parameter
            min_samples (int): DBSCAN min_samples parameter
        """
        self.input_file = input_file
        self.output_file = output_file
        self.max_event = max_event
        self.eps = eps
        self.min_samples = min_samples
        
        # Columns in the data file
        self.columns = ["a", "b", "c", "x", "y", "z", "tb", "q"]
        
        # Statistics counters
        self.total_points_original = 0
        self.total_points_cleaned = 0
        self.events_processed = 0
        
        # Seagull detection parameters
        self.z_padding = 100  # Fixed padding of ±100 as requested
        
    def load_data(self):
        """Load the data file."""
        try:
            # Read the data with space as delimiter, skipping header row
            self.df = pd.read_csv(self.input_file, sep=r'\s+', names=self.columns, skiprows=1)
            
            # Convert columns to appropriate types
            for col in ["a", "x", "y", "z", "tb", "q"]:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
            # Drop rows with missing values in essential columns
            self.df = self.df.dropna(subset=["a", "x", "y", "z"])
            
            # Convert event number to integer
            self.df["a"] = self.df["a"].astype(int)
            
            print(f"Loaded {len(self.df)} points from {self.input_file}")
            print(f"Found {len(self.df['a'].unique())} unique events")
            print(f"Event numbers range from {self.df['a'].min()} to {self.df['a'].max()}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def filter_data(self):
        """Filter the data and write to output file."""
        # Open output file for writing
        with open(self.output_file, "w") as out_file:
            # Write header to match visualization code's expected format (6 columns only)
            out_file.write("a           x         y          z       tb      q\n")
            
            # Process each event up to max_event
            unique_events = sorted(self.df["a"].unique())
            
            print(f"Processing events up to {self.max_event}...")
            print("Focusing on seagull-like patterns and main clusters...\n")
            
            for event_number in unique_events:
                # Only process events up to max_event (1000)
                if event_number > self.max_event:
                    break
                
                # Get data for this event
                event_data = self.df[self.df["a"] == event_number]
                self.total_points_original += len(event_data)
                
                # Skip if too few points
                if len(event_data) < 10:
                    continue
                
                # Apply seagull-focused filtering
                filtered_data = self.filter_seagull_event(event_data, event_number)
                
                # Write filtered data to output file if any points remain
                if len(filtered_data) > 0:
                    for _, row in filtered_data.iterrows():
                        # Handle potential NaN values safely
                        a_val = int(row['a'])
                        x_val = row['x']
                        y_val = row['y'] 
                        z_val = row['z']
                        tb_val = row['tb'] if not pd.isna(row['tb']) else 0.0
                        q_val = int(row['q']) if not pd.isna(row['q']) else 0
                        
                        # Write only the 6 columns your visualization expects
                        line = f"{a_val}  {x_val:.4f}  {y_val:.4f}  {z_val:.4f}  {tb_val:.2f}  {q_val}\n"
                        out_file.write(line)
                    
                    self.total_points_cleaned += len(filtered_data)
                    self.events_processed += 1
                
                # Print progress every 50 events for more detailed tracking
                if event_number % 50 == 0:
                    print(f"Processed event {event_number}")
        
        # Print statistics
        self.print_statistics()
    
    def filter_seagull_event(self, event_data, event_number):
        """
        Filter a single event focusing on seagull-like patterns and main clusters.
        
        Strategy:
        1. Apply initial clustering to identify all clusters
        2. Detect the main cluster (prioritizing seagull-like patterns)
        3. Calculate adaptive Z-range that covers the main cluster
        4. Keep points that are within this range and close to main clusters
        
        Parameters:
            event_data (DataFrame): Data for a single event
            event_number (int): Event number
            
        Returns:
            DataFrame: Filtered data focusing on seagull patterns and main clusters
        """
        # Extract coordinates
        X = event_data[["x", "y", "z"]].values
        
        # Step 1: Apply initial clustering to identify all clusters
        X_scaled = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X_scaled)
        labels = db.labels_
        
        # Step 2: Analyze clusters and detect main/seagull-like patterns
        main_cluster_info = self.detect_main_cluster(X, labels, event_number)
        
        if main_cluster_info is None:
            # If no good clusters found, return empty (remove this event)
            print(f"Event {event_number}: No suitable main cluster found - event removed")
            return pd.DataFrame()
        
        # Step 3: Calculate adaptive Z-range based on main cluster
        z_min, z_max = self.calculate_adaptive_z_range(X, labels, main_cluster_info)
        
        # Step 4: Filter points based on main cluster proximity and Z-range
        filtered_mask = self.create_seagull_filter_mask(X, labels, main_cluster_info, z_min, z_max)
        
        # Apply the filter
        filtered_data = event_data.iloc[filtered_mask]
        
        # Print filtering results
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        points_kept = np.sum(filtered_mask)
        reduction = (1 - points_kept / len(event_data)) * 100
        
        print(f"Event {event_number}: {len(event_data)} -> {points_kept} points "
              f"(Z-range: {z_min:.0f} to {z_max:.0f}, {n_clusters} clusters, {reduction:.1f}% noise removed)")
        
        return filtered_data
    
    def detect_main_cluster(self, X, labels, event_number):
        """
        Detect the main cluster, prioritizing seagull-like patterns.
        
        Seagull characteristics:
        - Moderate to large size
        - Relatively planar structure (low Z variance relative to X-Y spread)
        - Centered or near-centered in X-Y plane
        - Good connectivity (not too scattered)
        
        Parameters:
            X: Point coordinates
            labels: Cluster labels from DBSCAN
            event_number: Event number for debugging
            
        Returns:
            dict: Information about the main cluster, or None if no suitable cluster found
        """
        # Get unique cluster labels (excluding noise)
        unique_labels = [label for label in np.unique(labels) if label != -1]
        
        if len(unique_labels) == 0:
            return None
        
        cluster_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = X[cluster_mask]
            
            # Calculate cluster characteristics
            cluster_size = len(cluster_points)
            
            # Skip very small clusters
            if cluster_size < 5:
                continue
            
            # Calculate geometric properties
            x_range = np.max(cluster_points[:, 0]) - np.min(cluster_points[:, 0])
            y_range = np.max(cluster_points[:, 1]) - np.min(cluster_points[:, 1])
            z_range = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
            
            # Calculate center
            center = np.mean(cluster_points, axis=0)
            
            # Calculate spread in X-Y vs Z (seagulls are more planar)
            xy_spread = np.sqrt(x_range**2 + y_range**2)
            z_spread = z_range
            
            # Seagull-like scoring criteria
            size_score = min(cluster_size / 50.0, 1.0)  # Prefer moderate to large clusters
            
            # Planarity score (seagulls have high X-Y spread relative to Z spread)
            if z_spread > 0:
                planarity_score = min(xy_spread / z_spread / 10.0, 1.0)
            else:
                planarity_score = 1.0
            
            # Centrality score (prefer clusters near origin, but allow some flexibility)
            distance_from_origin = np.sqrt(center[0]**2 + center[1]**2)
            centrality_score = max(0, 1.0 - distance_from_origin / 100.0)
            
            # Density score (prefer well-connected clusters)
            if xy_spread > 0:
                density_score = min(cluster_size / (xy_spread * z_spread + 1), 1.0)
            else:
                density_score = 1.0
            
            # Combined seagull score
            seagull_score = (size_score * 0.3 + 
                           planarity_score * 0.3 + 
                           centrality_score * 0.2 + 
                           density_score * 0.2)
            
            cluster_info = {
                'label': label,
                'size': cluster_size,
                'center': center,
                'ranges': (x_range, y_range, z_range),
                'seagull_score': seagull_score,
                'points': cluster_points
            }
            
            cluster_scores.append(cluster_info)
        
        if not cluster_scores:
            return None
        
        # Sort by seagull score and return the best candidate
        cluster_scores.sort(key=lambda x: x['seagull_score'], reverse=True)
        best_cluster = cluster_scores[0]
        
        return best_cluster
    
    def calculate_adaptive_z_range(self, X, labels, main_cluster_info):
        """
        Calculate the adaptive Z-range based on the main cluster location.
        
        Parameters:
            X: All point coordinates
            labels: Cluster labels
            main_cluster_info: Information about the main cluster
            
        Returns:
            tuple: (z_min, z_max) for filtering
        """
        main_cluster_points = main_cluster_info['points']
        
        # Get Z-range of the main cluster
        cluster_z_min = np.min(main_cluster_points[:, 2])
        cluster_z_max = np.max(main_cluster_points[:, 2])
        
        # Apply fixed padding of ±100 as requested
        z_min = cluster_z_min - self.z_padding
        z_max = cluster_z_max + self.z_padding
        
        return z_min, z_max
    
    def create_seagull_filter_mask(self, X, labels, main_cluster_info, z_min, z_max):
        """
        Create a filter mask to keep points relevant to seagull classification.
        
        Keep points that are:
        1. Within the adaptive Z-range
        2. Part of any cluster within this Z-range (not isolated noise)
        3. Reasonably close to the main cluster center
        
        Parameters:
            X: Point coordinates
            labels: Cluster labels
            main_cluster_info: Main cluster information
            z_min, z_max: Z-range boundaries
            
        Returns:
            np.array: Boolean mask for filtering
        """
        # Start with Z-range filter
        z_mask = (X[:, 2] >= z_min) & (X[:, 2] <= z_max)
        
        # Filter for clustered points (not noise) within Z-range
        cluster_mask = labels != -1
        
        # Combine Z-range and cluster filters
        basic_mask = z_mask & cluster_mask
        
        # Additional proximity filter: keep points reasonably close to main cluster center
        main_center = main_cluster_info['center']
        
        # Calculate distances from main cluster center (more lenient in X-Y, stricter in Z)
        distances_xy = np.sqrt((X[:, 0] - main_center[0])**2 + (X[:, 1] - main_center[1])**2)
        distances_z = np.abs(X[:, 2] - main_center[2])
        
        # Define proximity thresholds (adaptive based on main cluster size)
        xy_threshold = max(50, main_cluster_info['ranges'][0] + main_cluster_info['ranges'][1])
        z_threshold = max(self.z_padding, main_cluster_info['ranges'][2] + 50)
        
        proximity_mask = (distances_xy <= xy_threshold) & (distances_z <= z_threshold)
        
        # Final mask combines all criteria
        final_mask = basic_mask & proximity_mask
        
        return final_mask
    
    def print_statistics(self):
        """Print statistics about the filtering process."""
        print("\n===== Seagull-Focused Filtering Results =====")
        print(f"Events processed: {self.events_processed}")
        print(f"Original points: {self.total_points_original}")
        print(f"Cleaned points: {self.total_points_cleaned}")
        
        if self.total_points_original > 0:
            reduction = (1 - self.total_points_cleaned / self.total_points_original) * 100
            print(f"Background noise removed: {reduction:.1f}%")
            
        print(f"Output saved to: {self.output_file}")
        
        # Debug: Show some sample event numbers that were processed
        if hasattr(self, 'df'):
            processed_events = sorted([e for e in self.df["a"].unique() if e <= self.max_event])[:20]
            print(f"Sample event numbers processed: {processed_events}")
            if processed_events:
                print(f"Event number range: {min(processed_events)} to {max(processed_events)}")
        
        print("=" * 45)
        print("\nData is now optimized for seagull vs non-seagull classification!")
        print("- Seagull patterns preserved with main cluster focus")
        print("- Background noise removed")
        print("- Adaptive Z-ranges applied per event")
        print("- Ready for machine learning classifier training")

def main():
    """Main function to run the seagull-focused event cleaner."""
    print("Seagull Event Data Cleaner")
    print("=" * 40)
    print("Objective: Prepare clean data for seagull vs non-seagull classification")
    print("Strategy: Focus on main clusters and seagull-like patterns")
    print()
    
    # Create cleaner instance optimized for seagull detection
    cleaner = SeagullEventCleaner(
        input_file="Data file.txt",
        output_file="Cleaned_Data.txt",
        max_event=1000,
        eps=0.3,          # DBSCAN parameter for cluster density
        min_samples=5     # Minimum points to form a cluster
    )
    
    # Load data and process
    if cleaner.load_data():
        cleaner.filter_data()
        print("\nSeagull-focused data cleaning completed successfully!")
        print("The cleaned data is optimized for training a classification model.")
        print("You can now use 'Cleaned_Data.txt' with your visualization tools")
        print("and proceed to train a seagull vs non-seagull classifier.")
    else:
        print("Failed to load the data file.")

if __name__ == "__main__":
    main()  