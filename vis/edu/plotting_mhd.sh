#!/usr/bin/env bash

export athena_path=/home1/07756/emgsma/codes/athena_z4c/vis/python/
export amr_py_path=/home1/07756/emgsma/codes/athena_z4c/vis/plotting/

export hdf5_dir=/scratch1/07756/emgsma/athena/bns/production/tests/SFHo_M135-135_40km_newfloorfix/output-0000/
export out_dir=/scratch1/07756/emgsma/athena/bns/production/tests/SFHo_M135-135_40km_newfloorfix/images/

./rho.sh &
#./press.sh
#./Y.sh
./Bcc1.sh
#./Bcc2.sh
#./Bcc3.sh
#./rho_xz.sh
#./Bcc2_xz.sh
./t.sh &
#./adm.psi4.sh &
#./con.sh &
