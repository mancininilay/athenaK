import subprocess
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

def run_plot(i):
    plot_slice_script = 'plot_slice.py'
    variable_to_plot = 'dens'
    output_format = 'png'

    input_file = f'../bin/tov.mhd_w_bcc.{i:05d}.bin'
    output_file = f'output_{i:05d}.{output_format}'

    cmd = f'python {plot_slice_script} {input_file} {variable_to_plot} {output_file} --notex'
    
    result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
    output_number = result.stdout.strip()
    return float(output_number)

n_processes = 8
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
    for output in executor.map(run_plot, range(1, 1201)):
        results.append(output)

# Create a linspace from 1 to 1200 for the x-axis
x_values = np.linspace(1, 1200, num=1200)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x_values, results,'-')
plt.xlabel('Index')
plt.ylabel('rho-max')
plt.legend()
plt.grid(True)
plt.show()