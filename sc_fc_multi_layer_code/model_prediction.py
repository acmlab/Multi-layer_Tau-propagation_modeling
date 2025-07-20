import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ClassifyGraphDataset
from network import MSClassifyNet
from wirings import wirings

MODEL_PATH = "**.pt"
PTID_DX_FILE = "**.csv"

def predict_with_model(model_path=MODEL_PATH, ptid_dx_file=PTID_DX_FILE):
    in_features, out_features, num_classes = 160, 160, 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Dataset paths
    inputs_dir = "**input"
    targets_dir = "**truth"
    sc_dir = "**SC"
    fc_dir = "**FC"

    # Output directory
    output_dir = './model_predictions'
    os.makedirs(output_dir, exist_ok=True)

    # Read PTID and DX info
    ptid_list, dx_list = None, None
    try:
        ptid_dx_data = pd.read_csv(ptid_dx_file)
        ptid_list = ptid_dx_data.iloc[:, 0].tolist()
        dx_list = ptid_dx_data.iloc[:, 1].tolist()
        print(f"Loaded {len(ptid_list)} PTID and DX entries.")
    except Exception as e:
        print(f"Warning: Failed to load PTID/DX info: {e}")

    # Model setup
    wiring = wirings.FullyConnected(in_features, out_features)
    model = MSClassifyNet(num_classes, in_features, wiring, out_features)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    # Dataset and loader
    dataset = ClassifyGraphDataset(inputs_dir, targets_dir, sc_dir, fc_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Results containers
    pred_list, target_list, usc_list, ufc_list, error_list, lambdaB_list = [], [], [], [], [], []

    with torch.no_grad():
        for i, (inputs, targets, sc, fc) in tqdm(enumerate(loader), total=len(loader), desc="Predicting"):
            if torch.isnan(inputs).any():
                print(f"Warning: Sample {i} contains NaNs. Skipped.")
                continue

            inputs, targets, sc, fc = map(lambda x: x.squeeze().to(device), (inputs, targets, sc, fc))
            outputs, usc, ufc, _, _, lambdaB = model(inputs, sc, fc)

            pred_np = outputs.cpu().numpy().squeeze()
            target_np = targets.cpu().numpy()
            usc_np = usc.cpu().numpy().squeeze()
            ufc_np = ufc.cpu().numpy().squeeze()
            lambdaB_np = lambdaB.cpu().numpy()
            error_np = np.abs(pred_np - target_np)

            pred_list.append(pred_np)
            target_list.append(target_np)
            usc_list.append(usc_np)
            ufc_list.append(ufc_np)
            error_list.append(error_np)
            lambdaB_list.append(lambdaB_np)

    # Convert to arrays and flatten if needed
    pred_arr = np.array(pred_list)
    target_arr = np.array(target_list)
    usc_arr = np.array(usc_list)
    ufc_arr = np.array(ufc_list)
    error_arr = np.array(error_list)
    lambdaB_arr = np.array(lambdaB_list)

    def reshape_if_needed(arr):
        return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr

    pred_arr, target_arr, usc_arr, ufc_arr, error_arr, lambdaB_arr = map(reshape_if_needed,
        (pred_arr, target_arr, usc_arr, ufc_arr, error_arr, lambdaB_arr))

    mean_error = np.mean(error_arr, axis=0)
    M, N = pred_arr.shape

    results = {
        'predictions': pred_arr,
        'targets': target_arr,
        'absolute_errors': error_arr,
        'usc': usc_arr,
        'ufc': ufc_arr,
        'lambdaB': lambdaB_arr
    }

    if ptid_list and dx_list:
        cn_idx = [i for i, dx in enumerate(dx_list) if dx in ['CN', 'SMC', 'EMCI']]
        ad_idx = [i for i, dx in enumerate(dx_list) if dx in ['LMCI', 'AD']]

        for key, data in results.items():
            df = pd.DataFrame(data).round(3)
            df.insert(0, 'PTID', ptid_list[:len(df)])
            df.to_csv(os.path.join(output_dir, f'{key}_with_PTID.csv'), index=False)

            if key in ['usc', 'ufc']:
                df.iloc[cn_idx].to_csv(os.path.join(output_dir, f'{key}_CN_group.csv'), index=False)
                df.iloc[ad_idx].to_csv(os.path.join(output_dir, f'{key}_AD_group.csv'), index=False)

        pd.DataFrame([mean_error]).round(3).to_csv(
            os.path.join(output_dir, 'mean_absolute_errors_1xN.csv'), index=False, header=False)

    else:
        filenames = {
            'predictions': 'predictions_MxN.csv',
            'absolute_errors': 'absolute_errors_MxN.csv',
            'usc': 'usc_MxN.csv',
            'ufc': 'ufc_MxN.csv',
            'lambdaB': 'lambdaB_MxN.csv',
            'targets': 'targets_MxN.csv'
        }
        for key, name in filenames.items():
            pd.DataFrame(results[key]).round(3).to_csv(os.path.join(output_dir, name), index=False, header=False)

        pd.DataFrame([mean_error]).round(3).to_csv(
            os.path.join(output_dir, 'mean_absolute_errors_1xN.csv'), index=False, header=False)

    print(f"\nPrediction completed. Results saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    predict_with_model()