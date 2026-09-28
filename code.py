import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import wfdb
from scipy.signal import butter, filtfilt, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# STEP 1: Data Acquisition (Smart Local/Online)
# ==========================================
def load_mit_bih_record(record_name='200', local_dir='./mitdb'):
    """
    Loads an ECG record from LOCAL directory first.
    If not found, automatically downloads the entire MIT-BIH database once.
    """
    # Check if local directory exists and has data
    local_record_path = os.path.join(local_dir, record_name)
    
    if not os.path.exists(local_record_path + '.dat'):
        print(f"[STEP 1] Local data not found at '{local_dir}'.")
        print(f"[STEP 1] Downloading MIT-BIH database (this happens only ONCE)...")
        print(f"[STEP 1] This may take a few minutes depending on your internet...")
        wfdb.dl_database('mitdb', dl_dir=local_dir)
        print(f"[STEP 1] Database saved to: {os.path.abspath(local_dir)}")
    else:
        print(f"[STEP 1] Loading record {record_name} from LOCAL directory: {local_dir}")
    
    # Read from local path (NO pn_dir -> reads locally)
    record = wfdb.rdrecord(local_record_path)
    annotation = wfdb.rdann(local_record_path, 'atr')
    
    fs = record.fs
    raw_signal = record.p_signal[:, 0]
    
    return fs, raw_signal, annotation.sample, annotation.symbol


# ==========================================
# STEP 2: Preprocessing (Bandpass Filtering)
# ==========================================
def apply_butterworth_filter(raw_signal, fs, lowcut=0.5, highcut=45.0, order=4):
    print(f"[STEP 2] Filtering signal between {lowcut}Hz and {highcut}Hz...")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, raw_signal)
    return filtered_signal


# ==========================================
# STEP 3: Beat Detection (QRS Localization)
# ==========================================
def process_annotations_and_peaks(ann_peaks, ann_labels):
    print("[STEP 3] Processing R-peaks and structuring medical labels...")
    valid_indices = []
    binary_labels = []
    for i, symbol in enumerate(ann_labels):
        if symbol == 'N':
            binary_labels.append(0)
            valid_indices.append(i)
        elif symbol in ['L', 'R', 'V', 'A', 'E', '!']:
            binary_labels.append(1)
            valid_indices.append(i)
    return ann_peaks[valid_indices], np.array(binary_labels)


# ==========================================
# STEP 4: Feature Extraction (Time & Frequency)
# ==========================================
def extract_heartbeat_features(filtered_signal, r_peaks, binary_labels, fs):
    print("[STEP 4] Segmenting waveforms and extracting features...")
    pre_r = int(0.25 * fs)
    post_r = int(0.38 * fs)
    feature_matrix = []
    final_labels = []
    rr_intervals = np.diff(r_peaks) / fs
    
    for i in range(1, len(r_peaks) - 1):
        r_idx = r_peaks[i]
        start_idx = r_idx - pre_r
        end_idx = r_idx + post_r
        if start_idx < 0 or end_idx > len(filtered_signal):
            continue
        beat_segment = filtered_signal[start_idx:end_idx]
        pre_rr = rr_intervals[i-1]
        post_rr = rr_intervals[i]
        beat_mean = np.mean(beat_segment)
        beat_std = np.std(beat_segment)
        beat_max = np.max(beat_segment)
        beat_min = np.min(beat_segment)
        freqs, psd = welch(beat_segment, fs=fs, nperseg=len(beat_segment))
        peak_freq = freqs[np.argmax(psd)]
        mean_psd = np.mean(psd)
        
        feature_matrix.append({
            'pre_rr': pre_rr, 'post_rr': post_rr,
            'beat_mean': beat_mean, 'beat_std': beat_std,
            'beat_max': beat_max, 'beat_min': beat_min,
            'peak_freq': peak_freq, 'mean_psd': mean_psd
        })
        final_labels.append(binary_labels[i])
        
    return pd.DataFrame(feature_matrix), np.array(final_labels)


# ==========================================
# STEP 5: Classification (Machine Learning)
# ==========================================
def train_svm_classifier(X, y):
    print("[STEP 5] Splitting dataset and training Support Vector Machine...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return y_test, y_pred


# ==========================================
# STEP 6: Evaluation (Performance Metrics)
# ==========================================
def evaluate_project_performance(y_test, y_pred):
    print("\n" + "="*50)
    print("STEP 6: FINAL PROJECT PERFORMANCE REPORT")
    print("="*50)
    print("\nStatistical Metrics Table:")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Arrhythmia (1)']))
    print("Displaying Confusion Matrix plot...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Arrhythmia'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('ECG Signal Classification - Confusion Matrix')
    plt.show()


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    if y_prob is None:
        y_prob = y_pred.astype(float)
    elif np.ndim(y_prob) > 1:
        y_prob = y_prob[:, 1]

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'RMSE': np.sqrt(np.mean((y_true.astype(float) - y_prob) ** 2))
    }


def plot_results_summary(summary_rows):
    df = pd.DataFrame(summary_rows)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'RMSE']
    values = df[metrics].to_numpy()
    labels = df['Condition'].tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 2]})
    x = np.arange(len(labels))
    width = 0.15
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

    for idx, metric in enumerate(metrics):
        offset = (idx - 2) * width
        ax1.bar(x + offset, values[:, idx], width=width, label=metric, color=colors[idx], alpha=0.85)

    ax1.set_title('Results Summary: Accuracy, Precision, Recall, F1 and RMSE', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Metric Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=9)

    table_data = [['Condition'] + metrics]
    for _, row in df.iterrows():
        table_data.append([row['Condition']] + [f"{row[m]:.3f}" for m in metrics])

    ax2.axis('off')
    table = ax2.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center', colWidths=[0.35] + [0.12] * len(metrics))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    plt.tight_layout()
    plt.show()


# ==========================================
# Noise Injection (Signals & Systems)
# ==========================================
def add_ecg_noise(clean_signal, fs, noise_type='white_gaussian', snr_db=10.0):
    signal_power = np.mean(clean_signal**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    t = np.arange(len(clean_signal)) / fs

    if noise_type == 'white_gaussian':
        noise = np.random.normal(0, noise_std, len(clean_signal))
    elif noise_type == 'powerline_50Hz':
        noise = np.sqrt(2 * noise_power) * np.sin(2 * np.pi * 50 * t)
    elif noise_type == 'baseline_wander':
        noise = np.sqrt(2 * noise_power) * np.sin(2 * np.pi * 0.2 * t)
    else:
        raise ValueError("Unsupported noise type")
    return clean_signal + noise


# ==========================================
# Visualization Functions (FIXED LAYOUT)
# ==========================================
def plot_signal_comparison(raw, filtered, fs, title="Raw vs Filtered ECG Signal", duration_sec=5):
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(t, raw[:n_samples], color='#d62728', linewidth=0.8)
    axes[0].set_title('Raw ECG Signal (Baseline Wander & High-Freq Noise)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, filtered[:n_samples], color='#2ca02c', linewidth=1)
    axes[1].set_title('Filtered ECG Signal (Butterworth Bandpass: 0.5-45 Hz)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.subplots_adjust(left=0.125, bottom=0.06, right=0.9, top=0.92, wspace=0.2, hspace=0.56)
    plt.show()


def plot_noise_impact(clean_signal, fs, noise_types, snr_db=10.0, duration_sec=3):
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    fig_height = 3.5 * (len(noise_types) + 1) 
    fig, axes = plt.subplots(len(noise_types)+1, 1, figsize=(14, fig_height))
    axes[0].plot(t, clean_signal[:n_samples], color='#2ca02c', linewidth=1.2)
    axes[0].set_title('Clean Filtered Signal (Reference)', fontsize=12, fontweight='bold', color='#2ca02c', pad=15)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].grid(True, alpha=0.4, linestyle='--')
    colors = ['#ff7f0e', '#1f77b4', '#9467bd']
    for idx, noise_type in enumerate(noise_types):
        noisy = add_ecg_noise(clean_signal, fs, noise_type=noise_type, snr_db=snr_db)
        axes[idx+1].plot(t, clean_signal[:n_samples], color='#2ca02c', linewidth=0.6, alpha=0.5, label='Clean')
        axes[idx+1].plot(t, noisy[:n_samples], color=colors[idx], linewidth=0.9, label=f'+ {noise_type}')
        axes[idx+1].set_title(f'{noise_type.upper()} Noise Added (SNR={snr_db}dB)', fontsize=12, fontweight='bold', color=colors[idx], pad=15)
        axes[idx+1].set_xlabel('Time (seconds)')
        axes[idx+1].set_ylabel('Amplitude (mV)')
        axes[idx+1].legend(loc='upper right', fontsize=9)
        axes[idx+1].grid(True, alpha=0.4, linestyle='--')
    fig.suptitle('Impact of Standard Biomedical Noises on ECG Signal', fontsize=16, fontweight='bold')
    plt.subplots_adjust(left=0.125, bottom=0.06, right=0.9, top=0.92, wspace=0.2, hspace=0.56) 
    plt.show()


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    print("=== STARTING ECG SIGNAL PROCESSING PIPELINE ===\n")
    
    # ✅ Changed: Now reads from LOCAL './mitdb' directory
    fs, raw_signal, raw_peaks, raw_labels = load_mit_bih_record(
        record_name='200', 
        local_dir='./mitdb'
    )
    clean_peaks, clean_labels = process_annotations_and_peaks(raw_peaks, raw_labels)
    
    print("\n[VISUALIZATION 1] Raw Signal -> Filtered (Denoised) Signal")
    filtered_signal = apply_butterworth_filter(raw_signal, fs, lowcut=0.5, highcut=45.0, order=4)
    plot_signal_comparison(raw_signal, filtered_signal, fs, title="ECG Denoising: Raw -> Butterworth Filtered")
    
    print("\n[VISUALIZATION 2] Adding Standard Noises to Clean Signal")
    noise_types = ['white_gaussian', 'powerline_50Hz', 'baseline_wander']
    plot_noise_impact(filtered_signal, fs, noise_types, snr_db=10.0)
    
    print("\n[STEP 4-6] Running Classification Pipeline on Clean Filtered Signal...")
    X, y = extract_heartbeat_features(filtered_signal, clean_peaks, clean_labels, fs)
    y_test, y_pred = train_svm_classifier(X, y)
    evaluate_project_performance(y_test, y_pred)
    
    # ========== COMPREHENSIVE NOISE IMPACT ANALYSIS ==========
    print("\n" + "="*70)
    print("[NOISE IMPACT ANALYSIS] Testing classifier robustness in 3 scenarios:")
    print("="*70)
    
    baseline_accuracy = 0.0
    results_generalization = []
    results_noisy_train = []
    summary_results = []
    
    print("\n>>> TRAINING ON CLEAN DATA...")
    X_clean, y_clean = extract_heartbeat_features(filtered_signal, clean_peaks, clean_labels, fs)
    X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
        X_clean, y_clean, test_size=0.25, random_state=42, stratify=y_clean
    )
    
    scaler_clean = StandardScaler()
    X_train_clean_scaled = scaler_clean.fit_transform(X_train_clean)
    X_test_clean_scaled = scaler_clean.transform(X_test_clean)
    
    model_clean = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model_clean.fit(X_train_clean_scaled, y_train_clean)
    
    # Test on Clean (Baseline)
    y_pred_clean = model_clean.predict(X_test_clean_scaled)
    baseline_metrics = compute_classification_metrics(y_test_clean, y_pred_clean, model_clean.predict_proba(X_test_clean_scaled)[:, 1])
    baseline_accuracy = baseline_metrics['Accuracy']
    summary_results.append({'Condition': 'Baseline (Clean->Clean)', **baseline_metrics})
    print(f"    Baseline (Clean->Clean): Accuracy={baseline_accuracy:.4f}")
    
    # Test on Noisy (Generalization)
    for noise_type in noise_types:
        test_signal = add_ecg_noise(filtered_signal, fs, noise_type=noise_type, snr_db=10.0)
        X_feat, y_labels = extract_heartbeat_features(test_signal, clean_peaks, clean_labels, fs)
        X_feat_scaled = scaler_clean.transform(X_feat)
        y_pred_noisy = model_clean.predict(X_feat_scaled)
        metrics = compute_classification_metrics(y_labels, y_pred_noisy, model_clean.predict_proba(X_feat_scaled)[:, 1])
        results_generalization.append({'Condition': noise_type, **metrics})
        summary_results.append({'Condition': f'Clean->{noise_type}', **metrics})
        print(f"    Generalization (Clean->{noise_type}): Accuracy={metrics['Accuracy']:.4f}")
    
    print("\n>>> TRAINING ON NOISY DATA (Separate models):")
    for noise_type in noise_types:
        noisy_signal = add_ecg_noise(filtered_signal, fs, noise_type=noise_type, snr_db=10.0)
        X_noisy, y_noisy = extract_heartbeat_features(noisy_signal, clean_peaks, clean_labels, fs)
        
        X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
            X_noisy, y_noisy, test_size=0.25, random_state=42, stratify=y_noisy
        )
        
        scaler_noisy = StandardScaler()
        X_train_noisy_scaled = scaler_noisy.fit_transform(X_train_noisy)
        X_test_noisy_scaled = scaler_noisy.transform(X_test_noisy)
        
        model_noisy = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        model_noisy.fit(X_train_noisy_scaled, y_train_noisy)
        
        y_pred_noisy = model_noisy.predict(X_test_noisy_scaled)
        metrics = compute_classification_metrics(y_test_noisy, y_pred_noisy, model_noisy.predict_proba(X_test_noisy_scaled)[:, 1])
        results_noisy_train.append({'Condition': noise_type, **metrics})
        summary_results.append({'Condition': f'Noisy Train ({noise_type})', **metrics})
        print(f"    Noisy-Train ({noise_type}->{noise_type}): Accuracy={metrics['Accuracy']:.4f}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS...")
    print("="*70)
    
    # Plot 1: Generalization Gap (WITH DISTINCT COLORS)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results_generalization))
    
    # Colors for: White Gaussian, Powerline 50Hz, Baseline Wander
    colors_gen = ['#ff7f0e', '#2ca02c', '#d62728'] 
    
    plt.bar(x, [row['Accuracy'] for row in results_generalization], color=colors_gen, alpha=0.8, label='Train Clean -> Test Noisy')
    plt.axhline(y=baseline_accuracy, color='#2ca02c', linestyle='--', linewidth=2, 
                label=f'Baseline (Clean->Clean): {baseline_accuracy:.3f}')
    
    plt.xticks(x, [r['Condition'].upper() for r in results_generalization])
    plt.ylabel('Classification Accuracy')
    plt.title('Generalization Gap: Model Trained on Clean Data', fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, row in enumerate(results_generalization):
        plt.text(i, row['Accuracy'] + 0.02, f'{row["Accuracy"]:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    # Plot 2: Overfitting Check
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results_noisy_train))
    plt.bar(x, [row['Accuracy'] for row in results_noisy_train], color=['#1f77b4', '#9467bd', '#d62728'], alpha=0.8)
    plt.axhline(y=baseline_accuracy, color='#2ca02c', linestyle='--', linewidth=2, 
                label=f'Baseline (Clean->Clean): {baseline_accuracy:.3f}')
    plt.xticks(x, [r['Condition'].upper() for r in results_noisy_train])
    plt.ylabel('Classification Accuracy')
    plt.title('Overfitting Check: Model Trained & Tested on Same Noise', fontsize=14, fontweight='bold')
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for i, row in enumerate(results_noisy_train):
        plt.text(i, row['Accuracy'] + 0.02, f'{row["Accuracy"]:.3f}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Comprehensive Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(noise_types))
    width = 0.35
    
    plt.bar(x - width/2, [row['Accuracy'] for row in results_generalization], width, 
            label='Train Clean -> Test Noisy', color='#ff7f0e', alpha=0.8)
    plt.bar(x + width/2, [row['Accuracy'] for row in results_noisy_train], width, 
            label='Train Noisy -> Test Same', color='#1f77b4', alpha=0.8)
    
    plt.axhline(y=baseline_accuracy, color='#2ca02c', linestyle='--', linewidth=2, 
                label=f'Baseline (Clean->Clean): {baseline_accuracy:.3f}')
    plt.xticks(x, [r.upper() for r in noise_types])
    plt.ylabel('Classification Accuracy', fontsize=12)
    plt.title('Comprehensive Noise Impact Analysis (SNR=10dB)', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0.0, 1.05)
    plt.legend(loc='lower left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, (row1, row2) in enumerate(zip(results_generalization, results_noisy_train)):
        plt.text(i - width/2, row1['Accuracy'] + 0.02, f'{row1["Accuracy"]:.3f}', ha='center', fontweight='bold', fontsize=9)
        plt.text(i + width/2, row2['Accuracy'] + 0.02, f'{row2["Accuracy"]:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    summary_df = pd.DataFrame(summary_results)
    print("\nSummary Metrics Table:")
    print(summary_df.to_string(index=False))
    plot_results_summary(summary_results)

    print("\n=== PIPELINE EXECUTION COMPLETED SUCCESSFULLY ===")
    