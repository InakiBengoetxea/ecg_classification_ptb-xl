"""
utils/robustness.py
-------------------
Robustness testing suite for evaluating ECG models under simulated clinical noise.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_noise_sigma(signal, snr_db):
    """
    Calculates the standard deviation of noise required to achieve a target SNR.
    """
    signal_power = np.mean(signal ** 2)
    if signal_power == 0:
        return 1e-4
    noise_power = signal_power / (10 ** (snr_db / 10))
    return np.sqrt(noise_power)

def inject_baseline_wander(ecg_data, snr_db, fs=100):
    """
    Simulates respiration-induced baseline wander using a low-frequency (0.5 Hz) drift.
    """
    noisy_data = np.zeros_like(ecg_data)
    num_samples, seq_len, num_leads = ecg_data.shape
    t = np.arange(seq_len) / fs
    
    for i in range(num_samples):
        for lead in range(num_leads):
            sigma = calculate_noise_sigma(ecg_data[i, :, lead], snr_db)
            # Generate low-frequency baseline drift
            wander = sigma * np.sqrt(2) * np.sin(2 * np.pi * 0.5 * t)
            noisy_data[i, :, lead] = ecg_data[i, :, lead] + wander
            
    return noisy_data

def inject_muscle_artifact(ecg_data, snr_db):
    """
    Simulates electromyogram (EMG) muscle artifacts using high-frequency Gaussian noise.
    """
    noisy_data = np.zeros_like(ecg_data)
    num_samples, seq_len, num_leads = ecg_data.shape
    
    for i in range(num_samples):
        for lead in range(num_leads):
            sigma = calculate_noise_sigma(ecg_data[i, :, lead], snr_db)
            noise = np.random.normal(0, sigma, size=seq_len)
            noisy_data[i, :, lead] = ecg_data[i, :, lead] + noise
            
    return noisy_data

def inject_power_line_interference(ecg_data, snr_db, fs=100, pli_freq=50):
    """
    Injects pure harmonic waves matching standard power line frequency grids (50 Hz).
    """
    noisy_data = np.zeros_like(ecg_data)
    num_samples, seq_len, num_leads = ecg_data.shape
    t = np.arange(seq_len) / fs
    
    for i in range(num_samples):
        for lead in range(num_leads):
            sigma = calculate_noise_sigma(ecg_data[i, :, lead], snr_db)
            pli = sigma * np.sqrt(2) * np.sin(2 * np.pi * pli_freq * t)
            noisy_data[i, :, lead] = ecg_data[i, :, lead] + pli
            
    return noisy_data

def run_comprehensive_stress_test(model, X_test, y_test, device, snr_levels=[24, 18, 12, 6, 0]):
    """Evaluates classification performance across targeted noise intensities."""
    model.eval()
    records = []
    
    print("\n--- Running Baseline Evaluation ---")
    with torch.no_grad():
        clean_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        all_preds = []
        for i in range(0, len(clean_tensor), 64):
            batch = clean_tensor[i : i + 64]
            
            # Shape correction: Transform from (Batch, 1000, 12) to (Batch, 12, 1000)
            if batch.ndim == 3 and batch.shape[1] == 1000 and batch.shape[2] == 12:
                batch = batch.permute(0, 2, 1)
                
            all_preds.append(torch.sigmoid(model(batch)).cpu().numpy())
        clean_preds = np.vstack(all_preds)
        
    base_auc = roc_auc_score(y_test, clean_preds, average='macro')
    records.append({'Noise Type': 'Clean', 'SNR_dB': 'Inf', 'Macro_AUC': base_auc})
    print(f"[BASELINE] Clean Data Macro AUC: {base_auc:.4f}")
    
    print("\n--- Running Stress Testing Loop ---")
    for snr in snr_levels:
        print(f"[PROCESSING] Evaluating stress limits at SNR: {snr} dB")
        
        variants = {
            'Baseline Wander': inject_baseline_wander(X_test.copy(), snr_db=snr),
            'Muscle Artifact (EMG)': inject_muscle_artifact(X_test.copy(), snr_db=snr),
            'Power Line (50Hz)': inject_power_line_interference(X_test.copy(), snr_db=snr)
        }
        
        for name, X_noisy in variants.items():
            with torch.no_grad():
                noisy_tensor = torch.tensor(X_noisy, dtype=torch.float32).to(device)
                all_noisy_preds = []
                for i in range(0, len(noisy_tensor), 64):
                    batch_noisy = noisy_tensor[i : i + 64]
                    
                    # Shape correction: Transform from (Batch, 1000, 12) to (Batch, 12, 1000)
                    if batch_noisy.ndim == 3 and batch_noisy.shape[1] == 1000 and batch_noisy.shape[2] == 12:
                        batch_noisy = batch_noisy.permute(0, 2, 1)
                        
                    all_noisy_preds.append(torch.sigmoid(model(batch_noisy)).cpu().numpy())
                noisy_preds = np.vstack(all_noisy_preds)
                
            noisy_auc = roc_auc_score(y_test, noisy_preds, average='macro')
            records.append({'Noise Type': name, 'SNR_dB': snr, 'Macro_AUC': noisy_auc})
            print(f"  -> {name:25s} | Macro AUC: {noisy_auc:.4f}")
            
    return pd.DataFrame(records)