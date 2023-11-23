
import subprocess
import concurrent.futures

def run_plot(i):
    plot_slice_script = 'plot_slice_stream.py'
    variable_to_plot = 'derived:rho'
    output_format = 'png'

    input_file = f'../bin/tov.mhd_w_bcc.{i:05d}.bin'
    output_file = f'output_{i:05d}.{output_format}'

    cmd = f'python {plot_slice_script} {input_file} {variable_to_plot} {output_file} -n log -c cubehelix -d 2 --vmin 1e-16 --vmax 0.001 --notex'
    
    subprocess.run(cmd, shell=True, check=True)
    print(f'Processed {input_file} into {output_file}')

# Number of files to process
n_files = 434

# Number of parallel processes
n_processes = 8  # Adjust this number based on your CPU

with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
    executor.map(run_plot, range(n_files))
