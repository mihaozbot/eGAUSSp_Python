import pandas as pd
from tabulate import tabulate

# Assuming your data is stored in these variables:
# average_samples_per_class_per_client, all_client_metrics, all_federated_metrics, all_client_clusters, all_federated_clusters
# and you have functions calculate_metrics_statistics and calculate_cluster_stats defined

# Create a DataFrame for the table
columns = ['Metric', 'Client 1', 'Client 2', 'Client 3', 'Client 4', 'Federated']
rows = []

# Add rows for each metric
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics:
    row = [metric.capitalize() + ' $\uparrow$']
    for client_metrics in all_client_metrics:
        mean, std = calculate_metrics_statistics(client_metrics)[metric]
        row.append(f'{mean:.2f} ± {std:.2f}')
    fed_mean, fed_std = avg_std_federated_metrics[metric]
    row.append(f'\bf{{{fed_mean:.2f} ± {fed_std:.2f}}}')
    rows.append(row)

# Add row for clusters
cluster_row = ['#Clusters $\downarrow$']
for client_clusters in all_client_clusters:
    avg_clusters, std_clusters = calculate_cluster_stats(client_clusters)
    cluster_row.append(f'{avg_clusters:.2f} ± {std_clusters:.2f}')
avg_fed_clusters, std_fed_clusters = calculate_cluster_stats(all_federated_clusters)
cluster_row.append(f'\bf{{{avg_fed_clusters:.2f} ± {std_fed_clusters:.2f}}}')
rows.append(cluster_row)

# Create DataFrame
df = pd.DataFrame(rows, columns=columns)

# Print the table
print(tabulate(df, headers='keys', tablefmt='latex', showindex=False))
