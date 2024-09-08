#! /usr/bin/env python

# Script for plotting 1D data from Athena++ .hst files.

# Run "plot_hst.py -h" for help.

# Python modules
import argparse
import numpy as np

# Athena++ modules
import athena_read


def running_average(t, x, dt_mean):
    """
    Smoothen a timeseries over a given timestep
    """
    dt = t[1]-t[0]
    N = 2*int(round(dt_mean/dt)) + 1
    ker = np.ones(N)/N
    return t[(N-1)//2:-(N-1)//2], np.convolve(x, ker, mode="valid")


# Main function
def main(**kwargs):

    # get input file and read data
    input_file = kwargs['input']
    data = athena_read.hst(input_file)

    # get variable names, check they are valid, and set x/y data
    variables = kwargs['variables']
    if variables not in data:
        print('Invalid input variable name, valid names are:')
        for key in data:
            print(key)
        raise RuntimeError

    y_vals = data[variables]
    x_vals = data["time"]
    last_200_points = y_vals[-200:]  # Extract the last 200 points from y_vals 
    print(np.size(last_200_points))
    avg_last_200 = np.mean(last_200_points)  # Calculate the average of these points
    std_dev_last_200 = np.std(last_200_points)  # Calculate the standard deviation of these points

    print("Average of the last 200 points:", avg_last_200)
    print("Standard deviation of the last 200 points:", std_dev_last_200)
    x_vals, y_vals = running_average(x_vals, y_vals, 100.0)



    # Load Python plotting modules
    output_file = kwargs['output']
    if output_file == 'show':
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.show()
    else:
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x_vals, y_vals)
        plt.savefig(output_file)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='name of input (hst) file')
    parser.add_argument('-o', '--output',
                        default='show',
                        help='image filename; omit to display to screen')
    parser.add_argument('-v', '--variables',
                        help='comma-separated list of variables to be plotted')

    args = parser.parse_args()
    main(**vars(args))



