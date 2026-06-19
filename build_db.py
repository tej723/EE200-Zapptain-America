import os
import glob
import pickle
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import maximum_filter

def get_hashes(filepath):
    # Read the .mp3 file
    audio, fs = librosa.load(filepath, sr=22050, mono=True)
    
    # 1. Compute Spectrogram
    f, t, Sxx = signal.spectrogram(audio, fs, nperseg=1024)
    
    # 2. Find Peaks (Constellation)
    local_max = maximum_filter(Sxx, size=20) == Sxx
    threshold = np.max(Sxx) * 0.05
    peaks = (local_max) & (Sxx > threshold)
    peak_freqs, peak_times = np.where(peaks)
    
    # 3. Create Hashes (Pairs of peaks)
    hashes = []
    for i in range(len(peak_freqs)):
        for j in range(1, 6): # Pair with the next 5 peaks
            if i + j < len(peak_freqs):
                f1 = peak_freqs[i]
                f2 = peak_freqs[i+j]
                t1 = peak_times[i]
                delta_t = peak_times[i+j] - t1
                hashes.append(((f1, f2, delta_t), t1))
    return hashes

# Dictionary to store our database
database = {}

print("Building database... This will take a few minutes!")
# Loop through all 50 songs in your database folder
for filepath in glob.glob("database/*.mp3"): 
    song_name = os.path.basename(filepath).replace('.mp3', '')
    print(f"Indexing: {song_name}")
    
    hashes = get_hashes(filepath)
    for h, t1 in hashes:
        if h not in database:
            database[h] = []
        database[h].append((song_name, t1))

# Save the finished database permanently
with open("song_database.pkl", "wb") as f:
    pickle.dump(database, f)
print("Done! Database saved to song_database.pkl")