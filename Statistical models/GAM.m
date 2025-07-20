%% ==== 1. Load Data ====
data = readtable('Tau_SUVR.xlsx');

num_samples = size(data, 1);
scan_age = data.AGE;
subject = categorical(data.PTID);
sex = categorical(data.PTGENDER);

% Extract node-wise measures (columns containing 'Node')
Measures = data{:, contains(data.Properties.VariableNames, 'Node')};
num_rois = size(Measures, 2);
unique_subjects = unique(subject);
num_subjects = numel(unique_subjects);

% Initialize outputs
age_term_all = NaN(num_samples, num_rois);
Pvalue_all = NaN(num_rois, 1);

%% ==== 2. Fit GAMs and Estimate Age Effect per ROI ====
fprintf('Fitting GAMs for %d ROIs...\n', num_rois);

for roi = 1:num_rois
    y = Measures(:, roi);
    tbl = table(scan_age, sex, subject, y, 'VariableNames', {'scan_age', 'sex', 'subject', 'Measure'});

    try
        % Full model with age
        gam_full = fitrgam(tbl, 'Measure ~ scan_age + sex + subject', ...
                           'CategoricalPredictors', {'sex', 'subject'});
        
        % Extract partial dependence for age
        age_effect = partialDependence(gam_full, 'scan_age', tbl);
        age_term_all(:, roi) = age_effect;

        % Cross-validated prediction error (SSE) for full model
        cv_full = crossval(gam_full, 'KFold', 5);
        y_pred_full = kfoldPredict(cv_full);
        SSE_full = sum((y_pred_full - y).^2);

        % Null model without age
        gam_null = fitrgam(tbl, 'Measure ~ sex + subject', ...
                           'CategoricalPredictors', {'sex', 'subject'});
        cv_null = crossval(gam_null, 'KFold', 5);
        y_pred_null = kfoldPredict(cv_null);
        SSE_null = sum((y_pred_null - y).^2);

        % Compute F-statistic for model comparison
        df1 = 1;  % Degrees of freedom for scan_age
        df2 = num_samples - num_rois - 1;
        F = ((SSE_null - SSE_full) / df1) / (SSE_full / df2);
        Pvalue_all(roi) = 1 - fcdf(F, df1, df2);

    catch ME
        warning('GAM fitting failed for ROI %d: %s', roi, ME.message);
        Pvalue_all(roi) = NaN;
    end
end

%% ==== 3. Save Results ====
out_dir = 'Results';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

writematrix(age_term_all, fullfile(out_dir, 'GAM_tau_age.csv'));
writematrix(Pvalue_all, fullfile(out_dir, 'Pvalue.csv'));

fprintf('✅ GAM analysis complete. Results saved to:\n');
fprintf('   ➤ %s\n', fullfile(out_dir, 'GAM_tau_age.csv'));
fprintf('   ➤ %s\n', fullfile(out_dir, 'Pvalue.csv'));
