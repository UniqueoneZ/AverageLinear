import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import rankdata
from tqdm import tqdm

# read the dataset
file_path = r'datasets\etth1.csv'
data = pd.read_csv(file_path)

# count the total timestamps
total_rows = len(data)

# select the corresponding dataset
rows_to_keep = 12 * 30 * 24 # Etth1, Etth2
# rows_to_keep = 12 * 30 * 24 * 4 # Ettm1, Ettm2
# rows_to_keep = int(total_rows * 0.7) # weather, electricity, traffic

# keep the training data
data_70_percent = data.iloc[:rows_to_keep]


# save the training dataset
output_file_path = r'datasets\etth107.csv'
data_70_percent.to_csv(output_file_path, index=False)

# define the boundary, channel's correlations with other channels are all below this value will be not be grouped.
linear_boundary = 0.8

file_path = r"datasets\ettm207.csv"
data = pd.read_csv(file_path).iloc[:, 1:]

#calculate the spearman correlations 
ranked_data = data.apply(rankdata)
corr_matrix = np.corrcoef(ranked_data, rowvar=False)
length = data.shape[1]

# filter the channels 
close_indices = [(i, j) for i in range(length) for j in range(i+1, length) if abs(corr_matrix[i, j]) > linear_boundary]

G = nx.Graph()
G.add_edges_from(close_indices)

# use LPA to group
partition = list(nx.algorithms.community.label_propagation_communities(G))
final_list = []
for i in range(len(partition)):
    final_list.append(list(partition[i]))
#elements to remove
print("elements to remove:", final_list)

array = np.arange(length)
partition = [list(community) for community in partition]
elements_to_remove = sum(partition, [])
filtered_array = [x for x in array if x not in elements_to_remove]
#elements to remove1
print("elements to remove1:", filtered_array)
