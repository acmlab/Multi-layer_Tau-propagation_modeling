%% ADNI
% ---- File Paths (Modify to your local directories) ----
fc_file   = "ADNI_results\all_FC_samples.csv";
sc_file   = "ADNI_results\all_SC_samples.csv";
map_file  = "region_labels_mapped.csv";
meta_file = "Tau_SUVR.xlsx";

% ---- 1. Load ROI-to-Lobe Mapping ----
Tmap = readtable(map_file,'VariableNamingRule','preserve');
Tmap.Properties.VariableNames = {'ROI','Lobe'};
lobes = unique(Tmap.Lobe, 'stable');
nLobes = numel(lobes);

% ---- 2. Load FC and SC Data ----
opts = detectImportOptions(fc_file); opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tfc = readtable(fc_file, opts); Tfc.Properties.VariableNames{1} = 'Subject';
fc_vals = Tfc{:, 2:end};

opts = detectImportOptions(sc_file); opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tsc = readtable(sc_file, opts); Tsc.Properties.VariableNames{1} = 'Subject';
sc_vals = Tsc{:, 2:end};

% ---- 3. Load Meta File and Keep the Last Visit per Subject ----
meta = readtable(meta_file);
if ~isdatetime(meta.EXAMDATE)
    meta.EXAMDATE = datetime(meta.EXAMDATE,'ConvertFrom','yyyymmdd');
end
meta = sortrows(meta, ["PTID", "EXAMDATE"]);
[~, ~, ic] = unique(meta.PTID, 'stable');
idx_last = splitapply(@(x)x(end), (1:height(meta))', ic);
meta = meta(idx_last, :);
meta.Subject = meta.PTID;

% ---- 4. Define Diagnostic Groups ----
N = height(meta);
dxg = strings(N, 1);
dxg(ismember(meta.DX, ["CN", "SMC", "EMCI"])) = "nonAD";
dxg(ismember(meta.DX, ["LMCI", "AD"])) = "AD";
meta.DXgroup = categorical(dxg);

% ---- 5. Match IDs ----
[common_id, idx_meta, idx_fc] = intersect(meta.Subject, Tfc.Subject, 'stable');
fc_vals = fc_vals(idx_fc, :);
sc_vals = sc_vals(idx_fc, :);
dxgroup = meta.DXgroup(idx_meta);
idx_AD = dxgroup == "AD";
idx_nonAD = dxgroup == "nonAD";

% ---- 6. Perform ROI-wise t-tests ----
nROI = size(fc_vals, 2);
[p_fc, t_fc, t_fc_all, p_sc, t_sc, t_sc_all] = deal(zeros(1, nROI));
for r = 1:nROI
    [~, p_fc(r), ~, stats_fc] = ttest2(fc_vals(idx_AD, r), fc_vals(idx_nonAD, r));
    t_fc_all(r) = stats_fc.tstat;
    t_fc(r) = stats_fc.tstat * (p_fc(r) < 0.05);

    [~, p_sc(r), ~, stats_sc] = ttest2(sc_vals(idx_AD, r), sc_vals(idx_nonAD, r));
    t_sc_all(r) = stats_sc.tstat;
    t_sc(r) = stats_sc.tstat * (p_sc(r) < 0.05);
end

% ---- 7. Save t-test Results ----
writematrix(p_fc', 'Gene_results/adni_p_fc.txt');
writematrix(t_fc', 'Gene_results/adni_t_fc.txt');
writematrix(p_sc', 'Gene_results/adni_p_sc.txt');
writematrix(t_sc', 'Gene_results/adni_t_sc.txt');

fprintf("✅ ADNI group comparison completed. Results saved to:\n");
fprintf(" - p_fc.txt / t_fc.txt\n");
fprintf(" - p_sc.txt / t_sc.txt\n");

% ---- 7.1 Direction of FC-SC Difference for Significant ROIs ----
sig_idx = (p_fc < 0.05) | (p_sc < 0.05);
fcsc_diff_dir = zeros(1, nROI);
for r = 1:nROI
    if sig_idx(r)
        diff_val = mean(fc_vals(:, r)) - mean(sc_vals(:, r));
        fcsc_diff_dir(r) = double(diff_val > 1) * 1 + double(diff_val < 1) * -1;
    end
end
writematrix(fcsc_diff_dir', 'Gene_results/adni_fc_sc.txt');

% ---- 7.2 Retain t-values Based on FC-SC Mean Difference Direction ----
t_fcsc_by_mean_diff = zeros(1, nROI);
for r = 1:nROI
    if sig_idx(r)
        diff_val = mean(fc_vals(:, r)) - mean(sc_vals(:, r));
        if diff_val > 0
            t_fcsc_by_mean_diff(r) = t_fc_all(r);
        elseif diff_val < 0
            t_fcsc_by_mean_diff(r) = t_sc_all(r);
        end
    end
end
n_sig_roi = sum(t_fcsc_by_mean_diff ~= 0);
fprintf("Total %d ROIs show group differences and FC-SC contrast.\n", n_sig_roi);
writematrix(t_fcsc_by_mean_diff', 'Gene_results/adni_t_by_fcsc_mean_diff_0.05.txt');


%% OASIS
% ---- File Paths (Modify to your local directories) ----
fc_file   = "OASIS_results\all_FC_samples.csv";
sc_file   = "OASIS_results\all_SC_samples.csv";
map_file  = "region_labels_mapped.csv";
meta_file = "OASIS_metadata.csv";

% ---- 1. Load ROI-to-Lobe Mapping ----
Tmap = readtable(map_file,'VariableNamingRule','preserve');
Tmap.Properties.VariableNames = {'ROI','Lobe'};
lobes = unique(Tmap.Lobe, 'stable');
nLobes = numel(lobes);

% ---- 2. Load FC and SC Data ----
opts = detectImportOptions(fc_file); opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tfc = readtable(fc_file, opts); Tfc.Properties.VariableNames{1} = 'Subject';
fc_vals = Tfc{:, 2:end};

opts = detectImportOptions(sc_file); opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tsc = readtable(sc_file, opts); Tsc.Properties.VariableNames{1} = 'Subject';
sc_vals = Tsc{:, 2:end};

% ---- 3. Load Meta and Keep First Visit per Subject ----
meta = readtable(meta_file);
[~, ia] = unique(meta.Subject_name, 'stable');
meta = meta(ia, :);
meta.Subject = meta.Subject_name;

% ---- 4. Define Diagnostic Groups ----
N = height(meta);
dxg = strings(N, 1);
dxg(meta.dx1 == "Cognitively normal") = "nonAD";
dxg(meta.dx1 ~= "Cognitively normal") = "AD";
meta.DXgroup = categorical(dxg);

% ---- 5. Match IDs ----
[common_id, idx_meta, idx_fc] = intersect(meta.Subject, Tfc.Subject, 'stable');
fc_vals = fc_vals(idx_fc, :);
sc_vals = sc_vals(idx_fc, :);
dxgroup = meta.DXgroup(idx_meta);
idx_AD = dxgroup == "AD";
idx_nonAD = dxgroup == "nonAD";

% ---- 6. Perform ROI-wise t-tests ----
nROI = size(fc_vals, 2);
[p_fc, t_fc, t_fc_all, p_sc, t_sc, t_sc_all] = deal(zeros(1, nROI));
for r = 1:nROI
    [~, p_fc(r), ~, stats_fc] = ttest2(fc_vals(idx_AD, r), fc_vals(idx_nonAD, r));
    t_fc_all(r) = stats_fc.tstat;
    t_fc(r) = stats_fc.tstat * (p_fc(r) < 0.05);

    [~, p_sc(r), ~, stats_sc] = ttest2(sc_vals(idx_AD, r), sc_vals(idx_nonAD, r));
    t_sc_all(r) = stats_sc.tstat;
    t_sc(r) = stats_sc.tstat * (p_sc(r) < 0.05);
end
fprintf("OASIS: %d SC-significant ROIs, %d FC-significant ROIs.\n", sum(t_sc~=0), sum(t_fc~=0));

% ---- 7. Save Results ----
writematrix(p_fc', 'Gene_results/0001oasis_p_fc_0.05.txt');
writematrix(t_fc', 'Gene_results/0001oasis_t_fc_0.05.txt');
writematrix(p_sc', 'Gene_results/0001oasis_p_sc_0.05.txt');
writematrix(t_sc', 'Gene_results/0001oasis_t_sc_0.05.txt');

% ---- 7.1 FC-SC Mean Difference Direction ----
sig_idx = (p_fc < 0.05) | (p_sc < 0.05);
fcsc_diff_dir = zeros(1, nROI);
for r = 1:nROI
    if sig_idx(r)
        diff_val = mean(fc_vals(:, r)) - mean(sc_vals(:, r));
        fcsc_diff_dir(r) = double(diff_val > 1) * 1 + double(diff_val < 1) * -1;
    end
end
writematrix(fcsc_diff_dir', 'Gene_results/0001oasis_fc_sc_0.01.txt');

% ---- 7.2 Retain t-values Based on Mean Difference ----
t_fcsc_by_mean_diff = zeros(1, nROI);
for r = 1:nROI
    if sig_idx(r)
        diff_val = mean(fc_vals(:, r)) - mean(sc_vals(:, r));
        if diff_val > 0
            t_fcsc_by_mean_diff(r) = t_fc_all(r);
        elseif diff_val < 0
            t_fcsc_by_mean_diff(r) = t_sc_all(r);
        end
    end
end
n_sig_roi = sum(t_fcsc_by_mean_diff ~= 0);
fprintf("OASIS: %d ROIs show FC or SC significance and FC-SC contrast.\n", n_sig_roi);
writematrix(t_fcsc_by_mean_diff', 'Gene_results/0001oasis_t_by_fcsc_mean_diff_0.05.txt');

