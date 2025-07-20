% ==== Setup ROI and File Paths ====
ROI_list = {'160'};  % List of ROIs to process

for r = 1:length(ROI_list)
    roi_id = ROI_list{r};
    
    % Define file paths
    group_conn_path = fullfile('Results', roi_id, 'FC.mat');           % FC or SC matrix
    tau_vector_path = fullfile('Results', roi_id, 'mean_Delta_Tau');   % Mean nodal tau change
    output_dir = fullfile('dResults', roi_id);                         % Output directory
    
    % Create output directory if not exists
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % ==== Load Data ====
    load(group_conn_path);   % Loads 'mean_matrix' (n × n)
    load(tau_vector_path);   % Loads 'mean_DeltaTau' (n × 1)
    
    G = mean_matrix;         % Connection matrix
    TVector = mean_DeltaTau; % Tau propagation rate
    [~, n] = size(G);        % Number of nodes

    % ==== Compute Mean Tau of Neighbors ====
    mean_tau_neighbor = nan(n,1);
    threshold = 0.2;  % Edge threshold (set as needed)

    for i = 1:n
        neighbors = find(G(:,i) > threshold);
        mean_tau_neighbor(i) = mean(TVector(neighbors), 'omitnan');
    end

    % ==== Fit Linear Regression ====
    mdl = fitlm(mean_tau_neighbor, TVector);

    % Extract adjusted R and p-value
    r_adj = sqrt(mdl.Rsquared.Adjusted) * sign(mdl.Coefficients.Estimate(2));
    p_val = mdl.Coefficients.pValue(2);

    % ==== Predict Fit and Confidence Interval ====
    x_fit = linspace(min(mean_tau_neighbor), max(mean_tau_neighbor), 100)';
    [y_fit, y_ci] = predict(mdl, x_fit, 'Alpha', 0.05);  % 95% CI

    % ==== Plot Scatter and Fit ====
    figure; hold on;
    scatter(mean_tau_neighbor, TVector, 50, [43/255, 100/255, 52/255], ...
        'filled', 'MarkerFaceAlpha', 0.7);

    % Fit line
    plot(x_fit, y_fit, 'r', 'LineWidth', 2.5);

    % Confidence interval band
    fill([x_fit; flipud(x_fit)], [y_ci(:,1); flipud(y_ci(:,2))], ...
        'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

    % Annotate R and P
    text(min(mean_tau_neighbor), max(TVector), ...
        sprintf('R = %.3f\nP = %.3f', r_adj, p_val), ...
        'FontSize', 12, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

    % Labels and formatting
    xlabel('Mean Neighbor FC (Threshold > 0.2)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Tau Propagation Rate', 'FontSize', 14, 'FontWeight', 'bold');
    title(sprintf('Linear Fit (ROI %s)', roi_id), 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'LineWidth', 1.5, 'FontSize', 12, 'FontWeight', 'bold');
    hold off;

    % ==== Save Results ====
    save(fullfile(output_dir, 'regression_stats.mat'), 'r_adj', 'p_val', 'mean_tau_neighbor');
    saveas(gcf, fullfile(output_dir, 'regression_plot.svg'));
end
