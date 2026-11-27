import numpy as np
import matplotlib.pyplot as plt

sr = 8000
duration = 20 
t = np.arange(0,duration,1/sr)

np.random.seed(42)
audio = 0.05 * np.random.randn(len(t))

whistle_times=[2,5,8,12,16]
for start in whistle_times:
    mask = (t >= start) & (t < start + 2)
    audio[mask] += 0.8 * np.sin(2 * np.pi * 2000 * t[mask])

win = int(0.1 * sr)
n = len(audio) // win 
loundness = np.array([np.sqrt(np.mean(audio[i*win:(i+1)*win]**2))for i in range(n)])
time_axis = np.arange(n) * 0.1

threshold = 0.3
is_loud = loundness > threshold

count = 0 
for i in range(1,n):
    if is_loud[i] and not is_loud[i - 1]:
        count += 1
        print(f"Whistle {count} at {time_axis[i]:.1f} sec")

print("total whistles:",count)

plt.plot(time_axis,loundness)
plt.axhline(threshold , color = "red", linestyle="--",label = "threshold")
plt.xlabel("time (sec)")
plt.ylabel("londness")
plt.title("cooker whistle counter")
plt.legend()
plt.show()

