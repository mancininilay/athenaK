
import subprocess
import concurrent.futures

def run_plot(i):
    plot_slice_script = 'plot_slice_nilay.py'
    variable_to_plot = 'derived:vr'
    output_format = 'png'

    input_file = f'../bin/tov.mhd_w_bcc.{i:05d}.bin'
    input_file2 = f'../bin/tov.adm.{i:05d}.bin'
    output_file = f'output_{i:05d}.{output_format}'

    cmd = f'python {plot_slice_script} {input_file} {variable_to_plot} {input_file2} derived:adm {output_file} --vmin=-1e-6 --vmax 1e-6 --x1_min -15 --x1_max 15 --x2_min -15 --x2_max 15 --notex'
    
    subprocess.run(cmd, shell=True, check=True)
    print(f'Processed {input_file} into {output_file}')


# Number of parallel processes
n_processes = 8  # Adjust this number based on your CPU

with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
    executor.map(run_plot, range(1,1200,10))
