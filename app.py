import streamlit as st
import pandas as pd
import numpy as np
import librosa
import pickle
from collections import Counter
from scipy import signal
from scipy.ndimage import maximum_filter

# --- THE FIX: Force Matplotlib to run 'headless' ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- ADD THIS RIGHT HERE ---
st.set_page_config(page_title="Zapp 'tain America", page_icon="🎶", layout="centered")

# ==========================================
# 1. LOAD THE DATABASE (Only once)
# ==========================================
@st.cache_resource
def load_db():
    with open("song_database.pkl", "rb") as f:
        return pickle.load(f)
song_database = load_db()

# ==========================================
# 2. CORE MATH FUNCTIONS
# ==========================================
def process_audio_and_match(audio_file):
    # 1. Read Audio using librosa (LIMIT TO 15 SECONDS TO PREVENT CRASH!)
    audio, fs = librosa.load(audio_file, sr=22050, mono=True, duration=25)

    # 2. Compute Spectrogram
    standard_window = 1024
    f, t, Sxx = signal.spectrogram(audio, fs, nperseg=standard_window)
    
    # 3. Find Constellation Peaks
    neighborhood_size = 20
    local_max = maximum_filter(Sxx, size=neighborhood_size) == Sxx
    threshold = np.max(Sxx) * 0.05
    peaks = (local_max) & (Sxx > threshold)
    peak_freqs, peak_times = np.where(peaks)

    # 4. Create the Spectrogram Plot
    fig_spectrogram, ax1 = plt.subplots(figsize=(10, 4))
    ax1.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma', alpha=0.5)
    ax1.scatter(t[peak_times], f[peak_freqs], color='cyan', edgecolors='white', s=20, label='Peaks')
    ax1.set_ylim(0, 4000)
    ax1.set_title("Audio Fingerprint (Spectrogram + Constellation)")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [sec]")
    ax1.legend()

    # 5. Database Matching Logic
    query_hashes = []
    for i in range(len(peak_freqs)):
        for j in range(1, 6):
            if i + j < len(peak_freqs):
                f1 = peak_freqs[i]
                f2 = peak_freqs[i+j]
                q_t1 = peak_times[i]
                delta_t = peak_times[i+j] - q_t1
                query_hashes.append(((f1, f2, delta_t), q_t1))

    matches = []
    for h, q_t1 in query_hashes:
        if h in song_database:
            for song_name, db_t1 in song_database[h]:
                offset = round(db_t1 - q_t1, 2)
                matches.append((song_name, offset))
                
  # 6. Find the winning song and create Histogram
    if matches:
        match_counts = Counter(matches)
        
        # most_common(1) returns [((song_name, offset), count)]
        # We extract the tuple AND the number of times it matched
        best_match_tuple, highest_count = match_counts.most_common(1)[0]
        
        # --- THE FIX: CONFIDENCE THRESHOLD ---
        # Real songs will have high counts (often 50+). 
        # Accidental noise will usually be under 5.
        threshold = 15 
        
        if highest_count >= threshold:
            predicted_song_name = best_match_tuple[0] 
            winning_offsets = [offset for (song, offset) in matches if song == predicted_song_name]
        else:
            predicted_song_name = "No Match Found (Confidence too low)"
            winning_offsets = []
            
    else:
        predicted_song_name = "No Match Found"
        winning_offsets = []

    fig_histogram, ax2 = plt.subplots(figsize=(8, 3))
    if winning_offsets:
        ax2.hist(winning_offsets, bins=30, color='green', edgecolor='black')
    ax2.set_title(f"Time-Offset Matches for {predicted_song_name}")
    ax2.set_xlabel("Time Offset [sec]")
    ax2.set_ylabel("Number of Matches")

    return predicted_song_name, fig_spectrogram, fig_histogram
# ==========================================
# 3. THE WEB APP INTERFACE (STREAMLIT)
# ==========================================
st.title("⚡ Zapp 'tain America - Magical Mystery Tune")
st.write("Upload an audio clip to identify the song!")

# --- OPTIONAL: Display the database contents on the UI ---
with st.expander("📚 View Song Database"):
    st.write("This app recognizes the following songs:")
    # Extract unique song names from your database dictionary
    all_songs = set()
    for song_list in song_database.values():
        for song_name, _ in song_list:
            all_songs.add(song_name)
    
    # Display them nicely
    for song in sorted(list(all_songs)):
        st.write(f"- {song}")
# ---------------------------------------------------------

tab1, tab2 = st.tabs(["Single-Clip Mode", "Batch Mode"])

# --- MODE 1: SINGLE-CLIP MODE ---
with tab1:
# ... (the rest of your code continues normally)
    st.header("Identify a Single Song")
    single_file = st.file_uploader("Upload a query clip", type=["wav", "mp3"], key="single")
    
    if single_file is not None:
        st.audio(single_file) 
        
        with st.spinner("Analyzing audio fingerprint..."):
            prediction, fig_spec, fig_hist = process_audio_and_match(single_file)
            
            # Use a massive success banner for the winner
            st.success(f"🎉 **Match Found:** {prediction}")
            
            # --- THE UI UPGRADE: Hide graphs in a dropdown ---
            with st.expander("🔬 View the Mathematical Fingerprint"):
                st.write("Here is the Spectrogram and Time-Offset data the algorithm used to find the match:")
                st.pyplot(fig_spec)
                st.pyplot(fig_hist)
            
            # Clear memory
            plt.close(fig_spec)
            plt.close(fig_hist)
            
# --- MODE 2: BATCH MODE ---
with tab2:
    st.header("Batch Process Multiple Clips")
    batch_files = st.file_uploader("Upload query clips", type=["wav", "mp3"], accept_multiple_files=True, key="batch")
    
    if batch_files and st.button("Run Batch Match"):
        results = []
        with st.spinner("Processing batch..."):
            for file in batch_files:
                pred, _, _ = process_audio_and_match(file)
                results.append({"filename": file.name, "prediction": pred})
        
        df = pd.DataFrame(results)
        csv_data = df.to_csv(index=False).encode('utf-8')
        
        st.success("Batch processing complete!")
        st.download_button(
            label="Download results.csv",
            data=csv_data,
            file_name='results.csv',
            mime='text/csv'
        )
