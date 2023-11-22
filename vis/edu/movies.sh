#!/usr/bin/env bash

direc='/scratch1/07756/emgsma/athena/bns/production/mhd/M135_135_SFHo_40km_B5e15_dfloor14_7refl/images/'

python3 movie_t.py $direc &
python3 movie_rho.py $direc &
#python3 movie_press.py $direc &
#python3 movie_Y.py $direc &
#python3 movie_Bcc1.py $direc &
#python3 movie_Bcc2.py $direc &
#python3 movie_Bcc3.py $direc &
#python3 movie_Bcc2_xz.py $direc &
#python3 movie_rho_xz.py $direc &
#python3 movie_psi4.py &
#python3 movie_H.py &

