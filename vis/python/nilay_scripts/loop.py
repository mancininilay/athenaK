import os
import subprocess

# Modify these variables as needed
plot_slice_script = 'plot_slice.py'  # Path to your plot_slice.py script
variable_to_plot = 'dens'  # Variable you want to plot
output_format = 'png'  # Output file format

for i in range(86):
    input_file = f'../bin/tov.mhd_w_bcc.{i:05d}.bin'
    output_file = f'output_{i:05d}.{output_format}'

    # Construct the command
    cmd = f'python {plot_slice_script} {input_file} {variable_to_plot} {output_file} -n log -c plasma -d 2 --vmin 1e-16 --vmax 0.0013 --notex'

    # Execute the command
    subprocess.run(cmd, shell=True, check=True)

    print(f'Processed {input_file} into {output_file}')
