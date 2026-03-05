import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, welch, tf2sos  # 添加 tf2sos
from scipy.stats import entropy, skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ----------------------- 统一预处理函数（严格遵循用户要求） -----------------------
def preprocess_segment(df_segment: pd.DataFrame, fs: int = 256) -> np.ndarray:
    """
    基础公开版 EEG 预处理：
    1) 数值化 + 线性插值
    2) 取 tp9 / af7 / af8 / tp10 四通道
    3) 四通道平均参考（CAR）
    4) 0.5-50 Hz 带通滤波
    5) 50 Hz 工频陷波
    """
    required_cols = ["tp9", "af7", "af8", "tp10"]
    missing = [c for c in required_cols if c not in df_segment.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 1) 拷贝并转数值，非数值转 NaN 后线性插值
    df = df_segment.copy()
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # 如果整列全空，插值后仍可能有 NaN，这里做兜底
    df[required_cols] = df[required_cols].fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # 2) 统一命名
    df_eeg = df[required_cols].rename(columns={
        "tp9": "RAW_TP9",
        "af7": "RAW_AF7",
        "af8": "RAW_AF8",
        "tp10": "RAW_TP10"
    })

    # 3) 四通道平均参考（Common Average Reference）
    data = df_eeg.to_numpy(dtype=np.float64)
    ref = np.mean(data, axis=1, keepdims=True)
    data = data - ref

    # 4) 带通滤波：0.5 - 50 Hz
    sos_bp = butter(
        N=4,
        Wn=[0.5, 50.0],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    data = sosfiltfilt(sos_bp, data, axis=0)

    # 5) 50 Hz 工频陷波
    b_notch, a_notch = iirnotch(w0=50, Q=30, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    data = sosfiltfilt(sos_notch, data, axis=0)

    df_processed = pd.DataFrame(
        data,
        columns=["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
    )
    sig = df_processed.to_numpy(dtype=np.float32)

    return sig

# ----------------------- 鲁棒特征提取（保持不变） -----------------------
def extract_robust_features(signal: np.ndarray, fs: int = 256) -> np.ndarray:
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    features = []

    for ch in range(2):
        f, psd = welch(signal[:, ch], fs=fs, nperseg=512, noverlap=256)
        abs_powers = []
        for low, high in bands.values():
            idx = (f >= low) & (f <= high)
            power = np.sum(psd[idx]) if np.any(idx) else 0.0
            abs_powers.append(power)
        total = np.sum(abs_powers) + 1e-8
        rel_powers = np.array(abs_powers) / total
        features.extend(rel_powers)

        stats = [
            np.std(signal[:, ch]),
            entropy(np.abs(signal[:, ch]) + 1e-8),
            skew(signal[:, ch]),
            kurtosis(signal[:, ch])
        ]
        features.extend(stats)

    # 前额不对称
    f, psd_af7 = welch(signal[:, 0], fs=fs, nperseg=512)
    f, psd_af8 = welch(signal[:, 1], fs=fs, nperseg=512)
    asymmetry = []
    for low, high in bands.values():
        idx = (f >= low) & (f <= high)
        asym = np.mean(np.log(psd_af8[idx] + 1e-8) - np.log(psd_af7[idx] + 1e-8))
        asymmetry.append(asym)
    features.extend(asymmetry)

    return np.array(features)

# ----------------------- 数据增强（仅训练时） -----------------------
def augment_signal(sig: np.ndarray, num_aug: int = 5) -> np.ndarray:
    augmented = [sig]
    std = np.std(sig, axis=0, keepdims=True)
    for _ in range(num_aug):
        noise = np.random.normal(0, np.random.uniform(0.02, 0.08) * std, sig.shape)
        scale = np.random.uniform(0.8, 1.2)
        shift = np.random.randint(-40, 40)
        aug1 = sig + noise
        aug2 = sig * scale
        aug3 = np.roll(sig, shift, axis=0)
        aug4 = (sig + noise) * scale
        augmented.extend([aug1, aug2, aug3, aug4])
    return np.array(augmented)

# ----------------------- 加载数据集 -----------------------
def load_dataset(data_dir: str, fs: int = 256, augment: bool = True):
    features_all = []
    labels_all = []
    window_size = fs * 4

    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        label = int(fname.split('_')[1].split('.')[0])
        if label not in [0, 1, 2]:
            continue

        df = pd.read_csv(os.path.join(data_dir, fname))
        df.columns = ["timestamp", "tp9", "af7", "af8", "tp10"]
        num_segments = len(df) // window_size

        for i in range(num_segments):
            start = i * window_size
            end = start + window_size
            seg = df.iloc[start:end]

            if len(seg) != window_size:
                continue

            sig = preprocess_segment(seg, fs)

            if augment:
                aug_sigs = augment_signal(sig, num_aug=5)
                for aug_sig in aug_sigs:
                    feat = extract_robust_features(aug_sig, fs)
                    features_all.append(feat)
                    labels_all.append(label)
            else:
                feat = extract_robust_features(sig, fs)
                features_all.append(feat)
                labels_all.append(label)

    X = np.array(features_all)
    y = np.array(labels_all)
    print(f"Loaded {X.shape[0]} samples (augment={augment}), feature dim: {X.shape[1]}")
    return X, y

# ----------------------- 模型训练/测试 -----------------------
def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, class_weight='balanced', random_state=42)),
            ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42))
        ], voting='soft'))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test=None):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    if y_test is not None:
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds))

    counts = np.bincount(preds, minlength=3)
    final_pred = int(np.argmax(counts))
    return {
        "preds": preds,
        "counts": counts.tolist(),
        "final_pred": final_pred,
        "probs_mean": probs.mean(axis=0).tolist()
    }

# ----------------------- 主函数 -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data_dir", type=str, default="ttt", help="Folder with zwl_0.csv, zwl_1.csv, zwl_2.csv")
    parser.add_argument("--input_csv", type=str, default="demo/unwear/upload_20260109_195349.csv", help="Single long raw CSV for demo mode")
    parser.add_argument("--model_path", type=str, default="model_ensemble.pkl")
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()

    if args.mode in ["train", "test"]:
        X, y = load_dataset(args.data_dir, args.fs, augment=(args.mode == "train"))

        if args.mode == "train":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
            model = train_model(X_train, y_train)
            joblib.dump(model, args.model_path)
            print(f"Model saved to {args.model_path}")
            evaluate_model(model, X_test, y_test)
        else:
            model = joblib.load(args.model_path)
            evaluate_model(model, X, y)

    else:  # demo
        if not args.input_csv:
            raise ValueError("--input_csv required for demo")
        model = joblib.load(args.model_path)

        df = pd.read_csv(args.input_csv)
        df.columns = ["timestamp", "tp9", "af7", "af8", "tp10"]
        window_size = args.fs * 4
        num_segments = len(df) // window_size
        print(f"Processing {num_segments} segments from {args.input_csv}")

        features = []
        for i in range(num_segments):
            start = i * window_size
            end = start + window_size
            seg = df.iloc[start:end]
            if len(seg) != window_size:
                continue
            sig = preprocess_segment(seg, args.fs)
            feat = extract_robust_features(sig, args.fs)
            features.append(feat)

        if not features:
            raise ValueError("No valid segments")
        feats = np.array(features)

        res = evaluate_model(model, feats)
        print("Per-segment predictions:", res["preds"].tolist())
        print(f"Final majority vote: {res['final_pred']} (counts: {res['counts']})")

if __name__ == "__main__":
    main()