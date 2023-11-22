from make_movie import make_video
import sys

direc = sys.argv[1]
print(direc)
outsufix = '.mp4'

make_video(direc + 'rho*abs*z=0*', direc + 'rho_xy' + outsufix, 20)
#make_video(direc + 'rho*abs*y=0*', direc + 'rho_xz' + outsufix, 20)
