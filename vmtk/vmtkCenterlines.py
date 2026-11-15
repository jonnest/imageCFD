import os
from vmtk import pypes

# NOTE
# use open/capped mesh at boundaries for extraction

dir_path = '/mnt/data/deepFlow/segmentations/'
folder = 'polar_tinnitus_240213/t240213_v1/'

file = 'openMesh.vtk'


vmtkcenterlines='vmtkcenterlines -seedselector profileidlist -sourceids 0 -targetids 1 -ifile '+ \
                os.path.join(dir_path,folder,file) + \
		        ' --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile ' + \
                os.path.join(dir_path,folder,file.split('.')[0]+'_cl.vtp')

myPype = pypes.PypeRun(vmtkcenterlines.replace('\\','/'))
 
