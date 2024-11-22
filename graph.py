import matplotlib.pyplot as plt

# Data
sizes = [64, 128, 256, 512, 1024]
fast_times = [0.0035114288330078125, 0.016477584838867188, 0.09692056973775227, 1.2479018370310466, 7.824625730514526]
gpu_times = [0.006440798441569011, 0.014967918395996094, 0.05311258633931478, 0.29705055554707843, 0.9703116416931152]

# Plot with customizations
plt.figure(figsize=(10, 6))
plt.plot(sizes, fast_times, marker="o", color="blue", label="Fast CPU")
plt.plot(sizes, gpu_times, marker="o", color="red", label="GPU")

# Labels and title
plt.xlabel("Input Size")
plt.ylabel("Runtime (seconds)")
plt.title("Matrix Multiplication Runtime Comparison: Fast CPU vs GPU Implementation")
plt.xlim(0, 1100)  # Limit x-axis from 0 to 1000
plt.ylim(0, 10)  # Limit y-axis from 0 to 10
plt.grid(True, which="both", linestyle="--", linewidth=0.75)
plt.legend()
plt.show()
