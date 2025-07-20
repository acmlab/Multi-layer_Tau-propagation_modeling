%% 0. Clean workspace
clear; clc; close all;

%% 1. File paths (adjust as needed)
fc_file   = "ADNI_results/all_FC_samples.csv";
sc_file   = "ADNI_results/all_SC_samples.csv";
map_file  = "region_labels_mapped.csv";
meta_file = "Tau_SUVR.xlsx";

%% 2. Read ROI→Lobe mapping
Tmap = readtable(map_file, 'VariableNamingRule','preserve');
Tmap.Properties.VariableNames = {'ROI','Lobe'};
lobes   = unique(Tmap.Lobe,'stable');
nLobes  = numel(lobes);

%% 3. Read FC data
opts = detectImportOptions(fc_file);
opts.VariableNamesLine = 1; opts.DataLine = 2;
opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tfc = readtable(fc_file, opts);
Tfc.Properties.VariableNames{1} = 'Subject';
fc_vals = Tfc{:,2:end};
nRegions = size(fc_vals, 2);

%% 4. Read SC data
opts = detectImportOptions(sc_file);
opts.VariableNamesLine = 1; opts.DataLine = 2;
opts.VariableTypes(1) = {'string'}; opts.VariableTypes(2:end) = {'double'};
Tsc = readtable(sc_file, opts);
Tsc.Properties.VariableNames{1} = 'Subject';
sc_vals = Tsc{:,2:end};

%% 5. Read metadata and keep last visit per subject
meta = readtable(meta_file);
if ~isdatetime(meta.EXAMDATE)
    meta.EXAMDATE = datetime(meta.EXAMDATE,'ConvertFrom','yyyymmdd');
end
meta = sortrows(meta, ["PTID","EXAMDATE"]);
[~,~,ic] = unique(meta.PTID,'stable');
idx_last = splitapply(@(x)x(end),(1:height(meta))',ic);
meta = meta(idx_last,:);
meta.Subject = meta.PTID;

%% 6. Build grouping variables
% DXgroup
dxg = repmat("nonAD", height(meta),1);
dxg(ismember(meta.DX,["LMCI","AD"])) = "AD";
meta.DXgroup = categorical(dxg);
% AGEgroup
ageg = repmat("60down", height(meta),1);
ageg(meta.AGE>=61 & meta.AGE<=75) = "61-75";
ageg(meta.AGE>75  & meta.AGE<=85) = "75-85";
ageg(meta.AGE>=85)                 = "85up";
meta.AGEgroup = categorical(ageg);
% APOE4group
apo = repmat("APOE40", height(meta),1);
apo(meta.APOE4~=0) = "APOE4";
meta.APOE4group = categorical(apo);
% SEXgroup
sex = repmat("Female", height(meta),1);
sex(meta.PTGENDER=="Male") = "Male";
meta.SEXgroup = categorical(sex);

%% 7. Numeric covariates
sex_num = double(meta.SEXgroup=="Male");      % 1=Male, 0=Female
apo4    = double(meta.APOE4group=="APOE4");   % 1=carrier, 0=non‑carrier

%% 8. Align metadata to FC (and then SC)
[found_fc, loc_fc] = ismember(Tfc.Subject, meta.Subject);
assert(all(found_fc), 'Missing FC subjects in meta');
meta2 = meta(loc_fc(found_fc), :);
fc2   = fc_vals(found_fc, :);
subs2 = Tfc.Subject(found_fc);

[found_sc, loc_sc] = ismember(subs2, Tsc.Subject);
assert(all(found_sc), 'Missing SC for some FC subjects');
sc2 = sc_vals(loc_sc, :);

%% 9. Pre‑allocate p‑value arrays
pvals_fc   = zeros(nRegions,1);
pvals_sc   = zeros(nRegions,1);
pvals_diff = zeros(nRegions,1);

%% 10. Loop over regions and fit GLMs
for j = 1:nRegions
    y_fc   = fc2(:,j);
    y_sc   = sc2(:,j);
    y_diff = y_sc - y_fc;

    % Build table (only use rows that exist in meta2)
    T = table( ...
        y_fc, y_sc, y_diff, ...
        sex_num(found_fc), apo4(found_fc), ...
        'VariableNames', {'y_fc','y_sc','y_diff','sex','apo4'} ...
    );
    T.Int = T.sex .* T.apo4;

    % FC model
    mdl = fitglm(T, 'y_fc ~ sex + apo4 + Int');
    p = mdl.Coefficients{'Int','pValue'};
    if ~isnan(p) && p<0.05, pvals_fc(j)=p; end

    % SC model
    mdl = fitglm(T, 'y_sc ~ sex + apo4 + Int');
    p = mdl.Coefficients{'Int','pValue'};
    if ~isnan(p) && p<0.05, pvals_sc(j)=p; end

    % SC–FC difference model
    mdl = fitglm(T, 'y_diff ~ sex + apo4 + Int');
    p = mdl.Coefficients{'Int','pValue'};
    if ~isnan(p) && p<0.05, pvals_diff(j)=p; end
end

%% 11. Save p‑values (one line per region)
writematrix(pvals_fc,   'interaction_pvals_FC.txt');
writematrix(pvals_sc,   'interaction_pvals_SC.txt');
writematrix(pvals_diff,'interaction_pvals_SCminusFC.txt');
