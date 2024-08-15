import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    print(f"Extracting features from {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def process_wav_file(wav_file, input_dir):
    file_path = os.path.join(input_dir, wav_file)
    return file_path, extract_features(file_path)

def main(input_dir, output_dir, num_clusters=5):
    # List all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} WAV files in {input_dir}")

    # Extract features for all wav files
    features = []
    file_paths = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_wav_file, wav_file, input_dir) for wav_file in wav_files]
        for future in futures:
            file_path, feature = future.result()
            file_paths.append(file_path)
            features.append(feature)

    # Convert features to numpy array
    features = np.array(features)
    print(f"Extracted features for {len(features)} files")

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Standardized the features")

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(features_scaled)
    print("Performed KMeans clustering")

    # Create subfolders and copy files
    for i in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{i}')
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_files = [file_paths[j] for j in range(len(file_paths)) if labels[j] == i]
        for file_path in cluster_files:
            shutil.copy(file_path, cluster_dir)
        print(f"Copied {len(cluster_files)} files to {cluster_dir}")

    print("Files have been grouped, copied, and folders renamed.")


def clean_folder(folder_path):
    """Clean the folder by deleting all its contents."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    input_dir = "C:\\AI\\datasets\\nate"  # Change to your input directory
    output_dir = "C:\\AI\\datasets\\nate\\sorted"  # Change to your output directory
    os.makedirs(output_dir, exist_ok=True)
    clean_folder(output_dir)
    num_clusters = 10  # Change to the desired number of clusters
    main(input_dir, output_dir, num_clusters)