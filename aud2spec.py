import sys
import numpy as np  
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.signal import spectrogram

# 1) pick a file: first CLI argument if given
wav_path = sys.argv[1] if len(sys.argv) > 1 else "example.wav"


# 2) read the WAV

sr, y = wavfile.read(wav_path);

# 3) make it float, mono, and normalized to [-1, 1]
y = y.astype(np.float32)
if y.ndim == 2:
    y = y.mean(axis=1)
y /= (np.max(np.abs(y)) + 1e-12)

# 4) compute spectrogram (Short-Time Fourier Transform)
f, t, S = spectrogram(
    y,
    fs=sr,
    window='hann',
    nperseg=1024,
    noverlap = 1024 - 256,
    mode='magnitude'
)

# 5) convert magnitude to decibels (log scale so quiet details show)
S_db = 20 * np.log10(S + 1e-12)

# 6) Plot it

plt.imshow(
    S_db,
    origin='lower',
    aspect='auto',
    extent=[t.min(), t.max(), f.min(), f.max()]
)
plt.colorbar(label='dB')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()
