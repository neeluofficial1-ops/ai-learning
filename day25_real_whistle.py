import numpy as np
import matplotlib.pyplot  as plt
from scipy.io import wavfile

rate , sound = wavfile.read("cooker.wav")
print("sample rate:" , rate)
print("shape:",sound.shape)

if sound.ndim == 2:
    sound = sound.mean(axis=1)

sound = sound / np.max(np.abs(sound))

window = int(rate * 0.1)
n = len(sound) // window
loud = []
for i in range(n):
    part = sound[i * window:(i +1)*window]
    loud.append(np.sqrt(np.mean(part ** 2)))
loud  = np.array(loud)
times = np.arange(n) * 0.1

print("total seconds:",round(len(sound) / rate , 1))
print("Max loudness:",round(loud.max(),3))
print("average loudeness:", round(loud.mean(),3))

threshold = 0.2
min_gap = 0

count = 0
last_time = -100
was_loud = False
for t , l in zip(times,loud):
    is_loud = l > threshold
    if is_loud and not was_loud and (t - last_time) > min_gap:
        count += 1
        last_time = t
        print(f"whistle {count} at {t:.1f} sec")
    was_loud = is_loud

print("total whistle:",count)

plt.plot(times , loud)
plt.axhline(threshold,color = "Red",linestyle="--")
plt.xlabel("times (sec)")
plt.ylabel("loudness (RMS)")
plt.title("real cooker sound")
plt.show()