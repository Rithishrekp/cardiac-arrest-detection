import os
import zipfile
import numpy as np
import pandas as pd
import scipy.signal as signal_lib
from scipy.signal import find_peaks

# Configure paths
raw_dir = r"c:\Users\Rithika\OneDrive\Pictures\Documents\cardiac_project\cardiac-arrest-detection\ml\datasets\raw"
processed_dir = r"c:\Users\Rithika\OneDrive\Pictures\Documents\cardiac_project\cardiac-arrest-detection\ml\datasets\processed"

mit_zip_path = os.path.join(raw_dir, "mit-bih-malignant-ventricular-ectopy-database-1.0.0.zip")
nor_zip_path = os.path.join(raw_dir, "norwegian-endurance-athlete-ecg-database-1.0.0 (1).zip")

mit_extract_dir = os.path.join(raw_dir, "mit_bih_temp")
nor_extract_dir = os.path.join(raw_dir, "norwegian_temp")

existing_dataset_path = os.path.join(processed_dir, "Integrated_Dataset_Final.csv")
new_dataset_path = os.path.join(processed_dir, "Integrated_Dataset_Final_v4.csv")

def unzip_dataset(zip_path, extract_to):
    print(f"Unzipping {os.path.basename(zip_path)} to {extract_to}...")
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print("Unzip complete.")

def detect_r_peaks(sig, fs, window_sec=10):
    """Estimate RR interval and heart rate using windowed peak detection."""
    window_size = int(window_sec * fs)
    all_peaks = []
    
    for i in range(0, len(sig), window_size):
        segment = sig[i:i+window_size]
        if len(segment) < 2:
            continue
        seg_min = np.min(segment)
        seg_max = np.max(segment)
        if seg_max == seg_min:
            continue
        norm_seg = (segment - seg_min) / (seg_max - seg_min)
        
        # R-peaks in a 10-second segment
        # distance = 0.4 seconds -> 0.4 * fs samples
        peaks, _ = find_peaks(norm_seg, distance=int(0.4 * fs), prominence=0.4)
        if len(peaks) < 2:
            peaks, _ = find_peaks(norm_seg, distance=int(0.4 * fs), prominence=0.2)
            
        for p in peaks:
            all_peaks.append(i + p)
            
    if len(all_peaks) < 2:
        return np.nan, np.nan
        
    all_peaks = np.sort(all_peaks)
    diffs = np.diff(all_peaks)
    valid_indices = diffs >= int(0.4 * fs)
    filtered_diffs = diffs[valid_indices]
    if len(filtered_diffs) == 0:
        return np.nan, np.nan
        
    rr_intervals_ms = (filtered_diffs / fs) * 1000.0
    mean_rr = np.mean(rr_intervals_ms)
    hr = 60000.0 / mean_rr if mean_rr > 0 else np.nan
    return mean_rr, hr

def process_record(record_path, source):
    """Load WFDB record, extract signal stats, RR-intervals, and Heart Rate."""
    import wfdb
    
    try:
        # Load signal and fields
        signals, fields = wfdb.rdsamp(record_path)
        fs = fields['fs']
        
        # Check signal shape
        if signals.size == 0 or len(signals.shape) < 2:
            print(f"Empty signal array in {record_path}")
            return None
            
        # Use first channel (Lead 0)
        sig = signals[:, 0]
        
        # Calculate numerical features
        sig_mean = np.mean(sig)
        sig_std = np.std(sig)
        sig_var = np.var(sig)
        sig_min = np.min(sig)
        sig_max = np.max(sig)
        sig_range = sig_max - sig_min
        rms = np.sqrt(np.mean(sig ** 2))
        sig_energy = np.sum(sig ** 2)
        
        # Perform windowed peak detection for both datasets
        mean_rr, hr = detect_r_peaks(sig, fs)
            
        return {
            "RecordID": os.path.basename(record_path),
            "_Source_": source,
            "HeartRate": hr,
            "RRInterval": mean_rr,
            "SignalMean": sig_mean,
            "SignalStd": sig_std,
            "SignalVar": sig_var,
            "SignalMin": sig_min,
            "SignalMax": sig_max,
            "SignalRange": sig_range,
            "RMS": rms,
            "SignalEnergy": sig_energy
        }
    except Exception as e:
        print(f"Error processing record {record_path}: {e}")
        return None

def process_extracted_dir(extract_dir, source):
    """Find all WFDB records in directory and process them."""
    records_processed = []
    
    # We walk the directory to locate .hea files (which denote WFDB records)
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.hea'):
                # Check for corresponding .dat file
                base_name = file[:-4]
                if base_name + '.dat' in files:
                    record_path = os.path.join(root, base_name)
                    # For windows/wfdb library paths, we should use absolute path
                    abs_record_path = os.path.abspath(record_path)
                    res = process_record(abs_record_path, source)
                    if res:
                        records_processed.append(res)
                        
    return records_processed

def main():
    # 1. Unzip the datasets
    unzip_dataset(mit_zip_path, mit_extract_dir)
    unzip_dataset(nor_zip_path, nor_extract_dir)
    
    # 2. Extract features from both datasets
    print("Processing MITBIH dataset...")
    mit_records = process_extracted_dir(mit_extract_dir, "MITBIH")
    print(f"Extracted {len(mit_records)} records from MITBIH.")
    
    print("Processing NorwegianECG dataset...")
    nor_records = process_extracted_dir(nor_extract_dir, "NorwegianECG")
    print(f"Extracted {len(nor_records)} records from NorwegianECG.")
    
    total_files = len(mit_records) + len(nor_records)
    print(f"Total files processed successfully: {total_files}")
    
    # 3. Create dataframe from new features
    new_features = mit_records + nor_records
    new_df = pd.DataFrame(new_features)
    print(f"Generated feature dataframe with shape: {new_df.shape}")
    
    # 4. Load existing dataset
    if os.path.exists(existing_dataset_path):
        existing_df = pd.read_csv(existing_dataset_path)
        shape_before = existing_df.shape
        print(f"Existing dataset loaded. Shape: {shape_before}")
    else:
        print(f"Error: Existing dataset not found at {existing_dataset_path}")
        return
        
    # 5. Merge datasets (concat)
    print("Merging datasets...")
    # Concat will stack rows. Existing columns will get NaN in new rows if not populated.
    # New feature columns (SignalMean, etc.) will get NaN in old rows.
    # Aligning columns: HeartRate, RRInterval, RecordID, _Source_ are in both dataframes.
    merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    shape_after = merged_df.shape
    print(f"Merged dataset shape: {shape_after}")
    
    # 6. Save new dataset
    merged_df.to_csv(new_dataset_path, index=False)
    print(f"Saved merged dataset to {new_dataset_path}")
    
    # 7. Print requested metrics and report
    print("\n" + "="*50)
    print("INTEGRATION METRICS REPORT")
    print("="*50)
    print(f"Number of ECG files processed:  {total_files}")
    print(f"Number of feature rows generated: {new_df.shape[0]}")
    print(f"Shape before integration:       {shape_before}")
    print(f"Shape after integration:        {shape_after}")
    print("\nSource-wise row counts (Shape after integration):")
    if "_Source_" in merged_df.columns:
        counts = merged_df["_Source_"].value_counts()
        for src, count in counts.items():
            print(f"  - {src}: {count} rows")
            
        print("\nVerification of source counts:")
        expected_sources = ["PF12RED", "ClassC", "PTB-XL", "SportDB2", "NorwegianECG", "MITBIH"]
        for src in expected_sources:
            cnt = counts.get(src, 0)
            print(f"  - {src} rows: {cnt}")
    else:
        print("Error: _Source_ column not found in merged dataset!")
    print("="*50 + "\n")

    # 8. Clean up temp directories
    print("Cleaning up temporary extraction directories...")
    import shutil
    if os.path.exists(mit_extract_dir):
        shutil.rmtree(mit_extract_dir)
    if os.path.exists(nor_extract_dir):
        shutil.rmtree(nor_extract_dir)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
