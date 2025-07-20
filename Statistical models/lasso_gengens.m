%% ==== 1. Load Data ====
[~, ~, rawData] = xlsread('160_gengens_name.xlsx');
geneNames = rawData(1, 2:end);
dataCells = rawData(2:end, 2:end);

% Convert cell array to numeric matrix X
X = zeros(size(dataCells));
for i = 1:numel(dataCells)
    if isnumeric(dataCells{i})
        X(i) = dataCells{i};
    else
        X(i) = str2double(dataCells{i});
    end
end

% Load p-values
y_pval = load("Gene_results/oasis_p_fc_0.05.txt"); % Replace with the desired file
% y_pval = load("Gene_results/adni_p_sc.txt");

% Ensure column vector
if size(y_pval,2) > size(y_pval,1)
    y_pval = y_pval';
end

% Convert to -log(p) and apply floor to avoid log(0)
epsilon = 1e-10;
y_pval(y_pval < epsilon) = epsilon;
y_all = -log(y_pval);

% Sparsify: keep only significant values
y = y_all;
p_thresh = 0.05; 
y(y_pval > p_thresh) = 0;


%% ==== 2. Set Parameters ====
num_iter = 100;  % Number of bootstrap iterations
[n, p] = size(X);
lambda_list = logspace(-3, 1, 20);  % Regularization parameters
num_lambda = length(lambda_list);

% Initialize selection matrix [lambda × gene × bootstrap]
select_matrix = false(num_lambda, p, num_iter);

%% ==== 3. Bootstrap LASSO Across Lambda Path ====
fprintf('Running %d bootstrap iterations across %d lambda values...\n', num_iter, num_lambda);
for k = 1:num_iter
    idx = randsample(n, n, true);  % Bootstrap sampling with replacement
    Xk = X(idx, :);
    yk = y(idx);

    [B, ~] = lasso(Xk, yk, 'Lambda', lambda_list);
    B(B < 0) = 0;  % Non-negativity constraint
    select_matrix(:, :, k) = (B > 0)';  % Convert to logical
end

%% ==== 4. Compute Selection Frequency ====
stability_freq = mean(select_matrix, 3);  % Average over bootstrap → [lambda × gene]

%% ==== 5. Visualization: LASSO Stability Path (Top Genes) ====
top = 10;  % Number of top genes to plot
mean_freq = mean(stability_freq, 1);
[~, topIdx] = maxk(mean_freq, top);
colors = lines(length(topIdx));

figure; hold on;
for i = 1:length(topIdx)
    plot(lambda_list, stability_freq(:, topIdx(i)), '-o', 'LineWidth', 1.5, 'Color', colors(i,:));
end
set(gca, 'XScale', 'log');
legend(geneNames(topIdx), 'Interpreter','none','Location','eastoutside');
xlabel('Lambda'); ylabel('Selection Frequency');
title('LASSO Stability Path (Top Genes)');

%% ==== 6. Save Full Stability Matrix ====
FreqTable = array2table(stability_freq, 'VariableNames', geneNames);
FreqTable.Lambda = lambda_list';
FreqTable = movevars(FreqTable, 'Lambda', 'Before', 1);
writetable(FreqTable, 'Results/Lasso_Stability_Path_FullADCN2.csv');

%% ==== 7. Optional: Bar Plot for Genes Above Frequency Threshold ====
freq_thresh = 0.197;
selectedIdx = find(mean_freq >= freq_thresh);
selectedGenes = geneNames(selectedIdx);
selectedFreq = mean_freq(selectedIdx);

[~, sortIdx] = sort(selectedFreq, 'descend');
sortedIdx = selectedIdx(sortIdx);  % Index in geneNames

% Uncomment below to plot
% figure;
% b = bar(selectedFreq(sortIdx), 'FaceColor', 'flat');
% for i = 1:length(sortIdx)
%     genePos = find(topIdx == sortedIdx(i));
%     if ~isempty(genePos)
%         b.CData(i,:) = colors(genePos,:);
%     end
% end
% set(gca, 'XTickLabel', selectedGenes(sortIdx), 'XTick', 1:length(sortIdx));
% xtickangle(45);
% ylabel('Selection Frequency');
% title(sprintf('Non-Negative LASSO Bootstrap (Freq ≥ %.2f)', freq_thresh));
% saveas(gcf, 'Results/Lasso_Bootstrap_SelectedGenes_BarADCN2.svg');

% fprintf('\n✅ Bar plot completed. Stability matrix and top genes saved.\n');


%% ==== 8. Save Top Gene Binary Selection Matrix and Frequency ====

% File 1: Binary matrix [bootstrap × top genes] for selection at any lambda
selected_binary_matrix = false(num_iter, length(topIdx));
for k = 1:num_iter
    sel_k = squeeze(select_matrix(:, topIdx, k));  % [lambda × top]
    selected_binary_matrix(k, :) = any(sel_k, 1);  % Binary across lambda
end

% save('Gene_results/oasis_sc_genes_binary_0.05.mat', 'selected_binary_matrix', 'topIdx', 'geneNames');

% File 2: Frequency matrix [lambda × top genes]
selected_freq_matrix = stability_freq(:, topIdx);
selected_gene_names = geneNames(topIdx);
lambda = lambda_list;

% save('Gene_results/oasis_sc_genes_freq_0.05.mat', 'selected_freq_matrix', 'selected_gene_names', 'lambda');

fprintf('\n✅ Saved two .mat files:\n');
fprintf('  ➤ selected_genes_binary.mat (100 × %d)\n', length(topIdx));
fprintf('  ➤ selected_genes_freq.mat (20 × %d)\n', length(topIdx));
