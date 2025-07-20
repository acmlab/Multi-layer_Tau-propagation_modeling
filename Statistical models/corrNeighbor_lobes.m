%% ==== 1. Load Data ====
ROI_id = '160';
conn_file = fullfile('Results', ROI_id, 'FC.mat');
tau_file  = fullfile('Results', ROI_id, 'TVector.txt');
label_dir = 'data';

% Load FC matrix
load(conn_file); % loads variable: mean_matrix
G = mean_matrix;

% Load nodal tau propagation vector
TVector = load(tau_file);  % [n × 1]

% Load region labels
load("FC_regions_label.mat");  % contains lobe_index_sc
load("SC_regions_label.mat");  % contains lobe_index

Yeo7_label = lobe_index_sc;  % 7 systems
Mesulam_label = lobe_index;  % 13 systems
n = size(G, 1);              % number of nodes

output_path = fullfile('Results', ROI_id);

%% ==== 2. Compute System-level Neighbor Means (Yeo7) ====
neighbor_matrix_Yeo = zeros(n, 7);  % [node × system]

for i = 1:n
    neighbors = find(G(:, i) > 0.2);  % threshold on FC
    neighbor_tau = TVector(neighbors);
    neighbor_labels = Yeo7_label(neighbors);

    for j = 1:7
        idx = neighbor_labels == j;
        neighbor_matrix_Yeo(i, j) = mean(neighbor_tau(idx), 'omitnan');
    end
end

% ==== Correlate Node Tau with Neighbor Tau by System ====
r_yeo = zeros(7);
p_yeo = ones(7);

for i = 1:7  % system of the seed node
    idx_i = Yeo7_label == i;
    T_i = TVector(idx_i);

    for j = 1:7  % system of neighbors
        N_ij = neighbor_matrix_Yeo(idx_i, j);
        if sum(~isnan(N_ij)) < 10
            continue;
        end
        [r, p] = corr(T_i, N_ij, 'rows', 'complete');
        n_eff = length(T_i) - sum(isnan(N_ij));
        adj_r2 = 1 - (1 - r^2) * (n_eff - 1) / (n_eff - 2);
        adj_r2 = max(adj_r2, 0);
        r_yeo(i, j) = sign(r) * sqrt(adj_r2);
        p_yeo(i, j) = p;
    end
end

% ==== FDR Correction ====
[p_fdr_flags, ~, p_adj_flat] = fdr_bh(p_yeo(:), 0.01, 'dep', 'yes');
p_yeo_FDR = reshape(p_adj_flat, size(p_yeo));
significant_mask = p_yeo_FDR < 0.05;
r_yeo_corrected = r_yeo .* significant_mask;
