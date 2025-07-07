import pandas as pd
import re
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque

def clean_vendor_name(vendor_name):
    """
    Clean a vendor name by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing common business suffixes
    4. Stripping extra whitespace
    
    Args:
        vendor_name (str): The original vendor name
        
    Returns:
        str: The cleaned vendor name
    """
    if pd.isna(vendor_name):
        return vendor_name
    
    # Convert to lowercase
    cleaned = vendor_name.lower()
    
    # Remove punctuation
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    
    # Define business suffixes to remove
    business_suffixes = [
        "inc", "inc.", "incorporated",
        "corp", "corp.", "corporation",
        "llc", "l.l.c.", "limited liability company",
        "ltd", "ltd.", "limited",
        "co", "co.", "company",
        "plc", "p.l.c.", "public limited company",
        "llp", "l.l.p.", "limited liability partnership",
        "gmbh", "gesellschaft mit beschränkter haftung",
        "ag", "aktiengesellschaft",
        "bv", "besloten vennootschap",
        "s.a.", "societe anonyme", "société anonyme",
        "sas", "société par actions simplifiée",
        "pty ltd", "proprietary limited",
        "n.v.", "naamloze vennootschap",
        "s.r.l.", "società a responsabilità limitata",
        "spa", "società per azioni",
        "oy", "osakeyhtiö",
        "as", "aksjeselskap"
    ]
    
    # Remove business suffixes (match at end of string)
    for suffix in business_suffixes:
        # Use word boundary to avoid removing partial matches
        pattern = r'\b' + re.escape(suffix) + r'\b\s*$'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Strip extra whitespace and normalize multiple spaces to single space
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def generate_embeddings(text_list, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Generate semantic embeddings for a list of text strings using sentence-transformers.
    
    Args:
        text_list (list): List of text strings to embed
        model_name (str): Name of the sentence-transformers model to use
        
    Returns:
        list: List of embedding vectors
    """
    print(f"Loading embedding model: {model_name}")
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Filter out NaN values and handle them
    valid_texts = []
    indices_map = []
    
    for i, text in enumerate(text_list):
        if pd.isna(text) or text == '':
            valid_texts.append("unknown")  # Use placeholder for empty/NaN values
        else:
            valid_texts.append(text)
        indices_map.append(i)
    
    print(f"Generating embeddings for {len(valid_texts)} text entries...")
    
    # Generate embeddings
    embeddings = model.encode(valid_texts, convert_to_tensor=False)
    
    # Convert to list of arrays for easier handling
    embeddings_list = [embedding.tolist() for embedding in embeddings]
    
    return embeddings_list

def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix for all embeddings.
    
    Args:
        embeddings (np.ndarray): Array of embeddings
        
    Returns:
        np.ndarray: Cosine similarity matrix
    """
    print("Computing cosine similarity matrix...")
    
    # Normalize embeddings for cosine similarity
    normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(normalized_embeddings)
    
    print(f"Similarity matrix computed: {similarity_matrix.shape}")
    
    return similarity_matrix

def find_nearest_neighbors_numpy(similarity_matrix, k=5):
    """
    Find k nearest neighbors for each vendor using numpy operations.
    
    Args:
        similarity_matrix (np.ndarray): Cosine similarity matrix
        k (int): Number of nearest neighbors to find
        
    Returns:
        tuple: (similarities, indices) arrays similar to FAISS output
    """
    print(f"Finding {k} nearest neighbors for each vendor...")
    
    n_vendors = similarity_matrix.shape[0]
    
    # Get top k+1 neighbors (including self) for each vendor
    # Use argsort and take the last k+1 elements (highest similarities)
    top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -(k+1):]
    
    # Reverse to get highest similarities first
    top_k_indices = np.flip(top_k_indices, axis=1)
    
    # Get corresponding similarity scores
    top_k_similarities = np.array([
        similarity_matrix[i, top_k_indices[i]] 
        for i in range(n_vendors)
    ])
    
    return top_k_similarities, top_k_indices

def create_clusters_from_neighbors(similarities, indices, similarity_threshold=0.85):
    """
    Create clusters based on mutual nearest neighbors and similarity threshold.
    
    Args:
        similarities (np.ndarray): Similarity scores from neighbor search
        indices (np.ndarray): Indices of nearest neighbors
        similarity_threshold (float): Minimum similarity for clustering
        
    Returns:
        tuple: (cluster_assignments, clusters)
    """
    print(f"Creating clusters with similarity threshold: {similarity_threshold}")
    
    n_vendors = len(indices)
    
    # Build adjacency graph based on mutual neighbors and similarity threshold
    adjacency = defaultdict(set)
    
    for i in range(n_vendors):
        # Skip self (first neighbor is always self with similarity ~1.0)
        for j, neighbor_idx in enumerate(indices[i][1:], 1):  # Skip index 0 (self)
            similarity = similarities[i][j]
            
            # Check if similarity meets threshold
            if similarity >= similarity_threshold:
                # Check if it's a mutual neighbor relationship
                # (neighbor_idx also has i in its neighbors)
                neighbor_neighbors = set(indices[neighbor_idx][1:])  # Skip self
                if i in neighbor_neighbors:
                    adjacency[i].add(neighbor_idx)
                    adjacency[neighbor_idx].add(i)
    
    # Find connected components using BFS
    visited = set()
    clusters = []
    cluster_id = 0
    cluster_assignments = [-1] * n_vendors  # -1 means no cluster assigned
    
    for i in range(n_vendors):
        if i not in visited:
            # Start new cluster
            cluster = []
            queue = deque([i])
            visited.add(i)
            
            while queue:
                current = queue.popleft()
                cluster.append(current)
                cluster_assignments[current] = cluster_id
                
                # Add unvisited neighbors to queue
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Only create cluster if it has more than 1 member
            if len(cluster) > 1:
                clusters.append(cluster)
                cluster_id += 1
            else:
                # Single vendor - assign to singleton cluster
                cluster_assignments[cluster[0]] = -1  # No cluster
    
    print(f"Created {len(clusters)} clusters from {n_vendors} vendors")
    
    return cluster_assignments, clusters

def process_vendor_csv_with_clustering(file_path, k=5, similarity_threshold=0.85):
    """
    Process a CSV file containing vendor names, generate embeddings, and perform clustering.
    
    Args:
        file_path (str): Path to the CSV file
        k (int): Number of nearest neighbors to consider
        similarity_threshold (float): Minimum similarity for clustering
        
    Returns:
        tuple: (DataFrame with clustering results, cluster information)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Verify the expected column exists
        if 'Vendor Name' not in df.columns:
            raise ValueError("CSV file must contain a 'Vendor Name' column")
        
        # Apply cleaning function to create new column
        df['Cleaned Name'] = df['Vendor Name'].apply(clean_vendor_name)
        
        # Generate embeddings for cleaned names
        embeddings = generate_embeddings(df['Cleaned Name'].tolist())
        
        # Add embeddings as a new column
        df['Embedding'] = embeddings
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(embeddings_array)
        
        # Find nearest neighbors
        similarities, indices = find_nearest_neighbors_numpy(similarity_matrix, k)
        
        # Create clusters
        cluster_assignments, clusters = create_clusters_from_neighbors(
            similarities, indices, similarity_threshold
        )
        
        # Add cluster assignments to dataframe
        df['Cluster ID'] = cluster_assignments
        
        # Prepare cluster information
        cluster_info = {
            'total_clusters': len(clusters),
            'clustered_vendors': sum(1 for cid in cluster_assignments if cid != -1),
            'singleton_vendors': sum(1 for cid in cluster_assignments if cid == -1),
            'clusters': clusters,
            'similarities': similarities,
            'indices': indices,
            'similarity_matrix': similarity_matrix
        }
        
        return df, cluster_info
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

def print_clustering_results(df, cluster_info):
    """
    Print detailed clustering results and examples.
    
    Args:
        df (pd.DataFrame): DataFrame with clustering results
        cluster_info (dict): Clustering information
    """
    print(f"\n{'='*60}")
    print("CLUSTERING RESULTS")
    print(f"{'='*60}")
    
    print(f"Total vendors: {len(df)}")
    print(f"Number of clusters: {cluster_info['total_clusters']}")
    print(f"Clustered vendors: {cluster_info['clustered_vendors']}")
    print(f"Singleton vendors: {cluster_info['singleton_vendors']}")
    
    if cluster_info['total_clusters'] > 0:
        print(f"\nFirst 5 clusters:")
        print("-" * 40)
        
        for i, cluster in enumerate(cluster_info['clusters'][:5]):
            print(f"\nCluster {i}:")
            for vendor_idx in cluster:
                vendor_name = df.iloc[vendor_idx]['Vendor Name']
                cleaned_name = df.iloc[vendor_idx]['Cleaned Name']
                print(f"  - '{vendor_name}' → '{cleaned_name}'")
                
            # Show similarities within cluster
            if len(cluster) > 1:
                print("  Similarities:")
                similarity_matrix = cluster_info['similarity_matrix']
                for j, vendor_idx in enumerate(cluster):
                    for k, other_idx in enumerate(cluster):
                        if vendor_idx < other_idx:  # Avoid duplicates
                            similarity = similarity_matrix[vendor_idx, other_idx]
                            vendor_name = df.iloc[vendor_idx]['Cleaned Name']
                            other_name = df.iloc[other_idx]['Cleaned Name']
                            print(f"    → {vendor_name} ↔ {other_name}: {similarity:.3f}")
    else:
        print("\nNo clusters found. All vendors are singletons.")
        print("Consider lowering the similarity threshold or increasing k.")

def print_embedding_info(df):
    """
    Print information about the embeddings.
    
    Args:
        df (pd.DataFrame): DataFrame containing embeddings
    """
    if 'Embedding' in df.columns and len(df) > 0:
        sample_embedding = df['Embedding'].iloc[0]
        print(f"\nEmbedding Information:")
        print(f"  - Embedding dimension: {len(sample_embedding)}")
        print(f"  - Embedding type: {type(sample_embedding)}")
        print(f"  - Sample embedding (first 5 values): {sample_embedding[:5]}")
        
        # Show embedding statistics
        all_embeddings = np.array(df['Embedding'].tolist())
        print(f"  - Mean embedding magnitude: {np.mean(np.linalg.norm(all_embeddings, axis=1)):.4f}")
        print(f"  - Embedding shape: {all_embeddings.shape}")

# Main execution
if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file_path = "vendor_names.csv"  # Update this path as needed
    
    # Clustering parameters
    k_neighbors = 5
    similarity_threshold = 0.85
    
    print("Processing vendor names with embeddings and clustering...")
    print(f"Parameters: k={k_neighbors}, similarity_threshold={similarity_threshold}")
    
    # Process the CSV file with clustering
    df, cluster_info = process_vendor_csv_with_clustering(
        csv_file_path, 
        k=k_neighbors, 
        similarity_threshold=similarity_threshold
    )
    
    if df is not None and cluster_info is not None:
        # Print basic information about the dataset
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Print embedding information
        print_embedding_info(df)
        
        # Print clustering results
        print_clustering_results(df, cluster_info)
        
        # Print the first 5 rows (excluding embeddings for readability)
        print(f"\n{'='*60}")
        print("SAMPLE DATA")
        print(f"{'='*60}")
        print("\nFirst 5 rows (without embeddings):")
        print(df[['Vendor Name', 'Cleaned Name', 'Cluster ID']].head())
        
        # Optional: Save the processed data
        output_file = "processed_vendors_with_clustering.csv"
        df.to_csv(output_file, index=False)
        print(f"\nProcessed data with clustering saved to: {output_file}")
        
        # Show cluster distribution
        cluster_counts = df['Cluster ID'].value_counts().sort_index()
        print(f"\nCluster distribution:")
        if -1 in cluster_counts.index:
            print(f"  Singletons (no cluster): {cluster_counts[-1]} vendors")
        for cluster_id in sorted([cid for cid in cluster_counts.index if cid >= 0]):
            print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} vendors")
            
    else:
        print("Failed to process the CSV file.") 