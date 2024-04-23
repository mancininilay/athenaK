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
x = []
indices = range(1, 1201, 10)  # Create a range of indices to iterate over
with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
    # Use zip to iterate over results and indices together
    for output, idx in zip(executor.map(run_plot, indices), indices):
        results.append(output)
        x.append(idx)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(x, results, '-')  # Use 'x' for x-axis values
plt.xlabel('Index')
plt.ylabel('rho-max')
plt.title('Plot of rho-max vs. Index')  # Added a title for clarity
plt.legend(['rho-max'])
plt.grid(True)
plt.savefig('rho-max_plot.png', format='png', dpi=300) 