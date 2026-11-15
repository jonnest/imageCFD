# from vmtk import pypes
# import os

# os.environ['VTK_USE_X'] = 'False'
# os.environ['VTK_USE_OFFSCREEN'] = 'True'

# print(dir(vmtk))

# pypes.PypeRun('vmtksurfacereader -ifile surface_0.vtp --pipe vmtkcenterlines --pipe vmtkrenderer --pipe vmtksurfaceviewer -opacity 0.25 
# --pipe vmtksurfaceviewer -i @vmtkcenterlines.o -array MaximumInscribedSphereRadius --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile centerline_test.vtp')

# Extracting the centerline
# pypes.PypeRun('vmtk vmtksurfacereader -ifile surface_0.vtp --pipe vmtkcenterlines --pipe vmtkcenterlineviewer array MaximumInscribedSphereRadius --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor')


### container vmtk
# pypes.PypeRun(f'vmtk vmtksurfacereader -ifile surface_0.stl --pipe vmtkcenterlines --pipe vmtkcenterlineviewer array MaximumInscribedSphereRadius --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor')

# pypes.PypeRun('vmtksurfacereader -ifile surface_0.stl --pipe vmtkcenterlines --pipe vmtkrenderer --pipe vmtksurfaceviewer -opacity 0.25 --pipe vmtksurfaceviewer -i @vmtkcenterlines.o -array MaximumInscribedSphereRadius --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile centerline_test.vtp')

'''
##########################################################################
import vmtk.vmtkscripts as vmtk
import os

os.environ['VTK_USE_X'] = 'False'
os.environ['VTK_USE_OFFSCREEN'] = 'True'


def extract_centerlines(input_filename, output_filename):
    # Create an instance of vmtkSurfaceReader to read the input geometry
    surface_reader = vmtk.vmtkSurfaceReader()
    surface_reader.InputFileName = input_filename
    surface_reader.Execute()

    # Create an instance of vmtkCenterlines to extract the centerlines
    centerlines = vmtk.vmtkCenterlines()
    centerlines.Surface = surface_reader.Surface
    centerlines.SeedSelectorName = 'openprofiles'
    centerlines.Execute()

    # Create an instance of vmtkSurfaceWriter to write the centerlines to a file
    centerline_writer = vmtk.vmtkSurfaceWriter()
    centerline_writer.Surface = centerlines.Centerlines
    centerline_writer.OutputFileName = output_filename
    centerline_writer.Execute()

# Example usage
input_filename = 'surface_0_0.0.stl'
output_filename = 'output_centerlines.vtp'
extract_centerlines(input_filename, output_filename)
'''

##########################################################################
import os
# import vmtk.pypes as pypes  
from vmtk import pypes

dir_path = '.'
file = 'surface_0_0.0.stl'

vmtkcenterlines='vmtkcenterlines -seedselector profileidlist -sourceids 0 -targetids 1 -ifile '+ os.path.join(dir_path,file) +\
		' --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile '+os.path.join(dir_path,file.split('.vtp')[0]+'_cl.vtp')
# myPype = pypes.PypeRun(vmtkbranchclipper.replace('\\','/'))
myPype = pypes.PypeRun(vmtkcenterlines.replace('\\','/'))
 
# pypes.PypeRun('vmtksurfacereader -ifile surface_0.stl --pipe vmtkcenterlines --pipe vmtkrenderer --pipe vmtksurfaceviewer -opacity 0.25 --pipe vmtksurfaceviewer -i @vmtkcenterlines.o -array MaximumInscribedSphereRadius --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile centerline_test.vtp')

