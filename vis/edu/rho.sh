#!/usr/bin/env bash

## running amr_output_err.py script with appropriate parameters.
## usage:
## set the ??? params and issue:
## $ bash con.H_example.par.sh

## where the hdf5 files are located:
#hdf5_dir="/pscratch/sd/e/exg5366/gr_bns_hydro/no_tau_flooring/output-0000"

## where to save the output
#out_dir="/pscratch/sd/e/exg5366/gr_bns_hydro/no_tau_flooring/output-0000/images"

## set athena path for import
#athena_path="/global/homes/e/exg5366/codes/athena_z4c/vis/python/"
## set path to the dir which has amr_output_err.py
#amr_py_path="/global/homes/e/exg5366/codes/athena_z4c/vis/plotting/"

## create ouput dir it if doesn't exist
if [[ ! -d ${out_dir} ]];
	then
		mkdir -p ${out_dir}
	fi

## setting python path to find athena reader
PYTHONPATH=:${PYTHONPATH}:${athena_path}:
export PYTHONPATH

#export hdf5_dir=/pscratch/sd/e/exg5366/gr_bns_mhd/newc2p/plm/output-0000/
#export out_dir=/pscratch/sd/e/exg5366/gr_bns_mhd/newc2p/plm/output-0000/images/

## run the script:
python3 "${amr_py_path}/amr_output_err.py" -i  "${hdf5_dir}" -o "${out_dir}" -e "64" \
-p "gr_bns.prim_xy" -s "2" -f "png" -c "z=0" -r "50" -a "plot" -g "0" -n "rho" --vmin=1e-10 --vmax=1e-3

#python3 "${amr_py_path}/amr_output_err.py" -i  "${hdf5_dir}" -o "${out_dir}" -e "64" \
#-p "gr_bns.prim_xz" -s "2" -f "png" -c "y=0" -r "50" -a "plot" -g "0" -n "rho" --vmin=1e-10 --vmax=1e-3

##########
## Help ##
##########

## -n flag can be any of these:

## list of fields for the output variable = z4c:
## z4c.chi z4c.gxx z4c.gxy z4c.gxz z4c.gyy z4c.gyz z4c.gzz z4c.Khat
## z4c.Axx z4c.Axy z4c.Axz z4c.Ayy z4c.Ayz z4c.Azz z4c.Gamx z4c.Gamy z4c.Gamz
## z4c.Theta z4c.alpha z4c.betax z4c.betay z4c.betaz

## list of field for the output variable = con:
## con.C con.H con.M con.Z con.Mx con.My con.Mz

## list of field for the output variable = adm:
## adm.gxx adm.gxy adm.gxz adm.gyy adm.gyz adm.gzz
## adm.Kxx adm.Kxy adm.Kxz adm.Kyy adm.Kyz adm.Kzz adm.psi4

## module you may need these on an HPC (python and numpy are assumed that exist):
## $ python3 -m pip install --upgrade findiff
## $ python3 -m pip install --upgrade --user findiff
## $ python3 -m pip install --upgrade --user regex
## $ python3 -m pip install --upgrade --user pyparsing
## $ python3 -m pip install --upgrade --user h5py
