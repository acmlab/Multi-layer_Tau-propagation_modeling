%% CN/AD Group Analysis
% Read data
data = readtable('Tau_SUVR.xlsx');

% Filter CN and AD groups
validGroups = {'CN', 'AD'};
data = data(ismember(data.DX, validGroups), :);

% Store first two time points for each subject
uniqueSubjects = unique(data.PTID);
tau_change_tbl = [];

for i = 1:length(uniqueSubjects)
    subID = uniqueSubjects{i};
    subData = data(strcmp(data.PTID, subID), :);
    if height(subData) < 2
        continue;
    end
    subData = sortrows(subData, 'EXAMDATE');
    subData = subData(1:2, :);
    tau_t1 = subData{1, contains(subData.Properties.VariableNames, 'Node')};
    tau_t2 = subData{2, contains(subData.Properties.VariableNames, 'Node')};
    age_t1 = subData.AGE(1);
    age_t2 = subData.AGE(2);
    delta_tau = (tau_t2 - tau_t1) ./ (age_t2 - age_t1);
    tau_change_tbl = [tau_change_tbl; {subID, subData.DX{1}, subData.PTGENDER{1}, delta_tau}];
end

tau_change_tbl = cell2table(tau_change_tbl, 'VariableNames', {'PTID', 'DX', 'Gender', 'DeltaTau'});

CN_idx = strcmp(tau_change_tbl.DX, 'CN');
AD_idx = strcmp(tau_change_tbl.DX, 'AD');
CN_tau = mean(cat(1, tau_change_tbl.DeltaTau(CN_idx)), 1);
AD_tau = mean(cat(1, tau_change_tbl.DeltaTau(AD_idx)), 1);
tau_change_rate = (AD_tau - CN_tau) ./ CN_tau;

% Statistical test using linear mixed model
p_values = zeros(1, 160);
for i = 1:160
    tau_vals = tau_change_tbl.DeltaTau;
    tau_vals = tau_vals(:, i);
    tbl = table(tau_change_tbl.PTID, strcmp(tau_change_tbl.DX, 'AD'), strcmp(tau_change_tbl.Gender, 'Female'), tau_vals, 'VariableNames', {'ID', 'Group', 'Sex', 'TauChange'});
    lme = fitlme(tbl, 'TauChange ~ Group + Sex + (1|ID)');
    p_values(i) = coefTest(lme);
end

alpha = 5e-5;
significant_regions = find(p_values < alpha);
disp('Significant brain regions with tau change rate differences:');
disp(significant_regions);

%% Early vs Late Stage Analysis
% Read data
data = readtable('F:\ACMLab_Data\ADNI-data_tau_amyloid_FDG_CT_info\amyloid_tau_FDG_CT\Tau_SUVR_Swapped_update_age.xlsx');

% Define stage groups
earlyStage = {'CN', 'SMC', 'EMCI'};
lateStage = {'LMCI', 'AD'};
data.Group = repmat({''}, height(data), 1);
data.Group(ismember(data.DX, earlyStage)) = {'EarlyStage'};
data.Group(ismember(data.DX, lateStage)) = {'LateStage'};
data.Group = string(data.Group);
data = data(ismember(data.Group, ["EarlyStage", "LateStage"]), :);

% Store first two time points
uniqueSubjects = unique(data.PTID);
tau_change_tbl = [];
for i = 1:length(uniqueSubjects)
    subID = uniqueSubjects{i};
    subData = data(strcmp(data.PTID, subID), :);
    if height(subData) < 2
        continue;
    end
    subData = sortrows(subData, 'EXAMDATE');
    subData = subData(1:2, :);
    tau_t1 = subData{1, contains(subData.Properties.VariableNames, 'Node')};
    tau_t2 = subData{2, contains(subData.Properties.VariableNames, 'Node')};
    age_t1 = subData.AGE(1);
    age_t2 = subData.AGE(2);
    delta_tau = (tau_t2 - tau_t1) / (age_t2 - age_t1);
    tau_change_tbl = [tau_change_tbl; {subID, subData.DX{1}, subData.PTGENDER{1}, delta_tau, subData.Group{1}}];
end

tau_change_tbl = cell2table(tau_change_tbl, 'VariableNames', {'PTID', 'DX', 'Gender', 'DeltaTau', 'Group'});

early_tau = mean(cat(1, tau_change_tbl.DeltaTau(strcmp(tau_change_tbl.Group, 'EarlyStage'))), 1);
late_tau = mean(cat(1, tau_change_tbl.DeltaTau(strcmp(tau_change_tbl.Group, 'LateStage'))), 1);

% Statistical test with LME
p_values = zeros(1, 160);
t_values = zeros(1, 160);
for i = 1:160
    tau_vals = tau_change_tbl.DeltaTau;
    tau_vals = tau_vals(:, i);
    tbl = table(tau_change_tbl.PTID, strcmp(tau_change_tbl.DX, 'AD'), strcmp(tau_change_tbl.Gender, 'Female'), tau_vals, 'VariableNames', {'ID', 'Group', 'Sex', 'TauChange'});
    lme = fitlme(tbl, 'TauChange ~ Group + Sex + (1|ID)');
    p_values(i) = coefTest(lme);
    coeffs = lme.Coefficients;
    t_values(i) = coeffs.tStat(2);
end

alpha = 5e-5;
significant_regions = find(p_values < alpha);
disp('Significant brain regions with tau change rate differences:');
disp(significant_regions);
disp(table(significant_regions', t_values(significant_regions)', p_values(significant_regions)', 'VariableNames', {'BrainRegion', 'TValue', 'PValue'}));