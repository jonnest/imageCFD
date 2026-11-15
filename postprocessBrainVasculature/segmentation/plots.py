import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt



def plotScalarOnShape(msh_file,scalar,scalar_name,save_file,*arg):
    '''
    Plots scalar value on surface of geometry
    
    Parameters
    ----------
    msh_file : '<name>.stl'
        plotted geometry file.
    scalar : (N, ) numpy nd array
        scalar value on vertices.
    scalar_name : string
        name of scalar.
    save_file : string
        file name with ending ['.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff']

    Returns
    -------
    None.

    '''
    mesh = pv.read(msh_file)
    #sargs = dict(interactive=True)
    sargs = dict(height=0.7, vertical=True, 
                 position_x=0.85, position_y=0.05,
                 title=scalar_name,
                 title_font_size=20,
                 label_font_size=16,
                 )

    #annotations = {
    #    0.75*np.max(dist_rel): "High",
    #    0.1*np.max(dist_rel): "Low",
    #}

    pv.global_theme.background = 'white'
    pv.global_theme.font.color = 'black'
    
    my_theme = pv.themes.DocumentTheme()
    my_theme.cmap = 'jet'
    my_theme.show_edges = False
    pv.global_theme.load_theme(my_theme)


    pl = pv.Plotter()
    mesh["scalars"] = scalar
    #mesh.plot(scalar_bar_args=sargs)
    pl.add_mesh(mesh, show_scalar_bar = False)
    pl.add_scalar_bar(height=0.7, vertical=True, 
                 position_x=0.84, position_y=0.05,
                 title=scalar_name,
                 title_font_size=40,
                 label_font_size=40)
    
    pl.view_xz()
    pl.camera.roll += 180       # turn around x-axis
    pl.camera.azimuth += 20     # turn around z-axis
    pl.camera.elevation += 30   # turn around y-axis
    if arg!=():
        pl.add_title(title=arg[0])
        
    pl.show(screenshot=save_file, window_size=[1250,1200]) 
    
              #, annotations=annotations)
              
    #mesh.save('ventricle.ply') # .ply .stl .vtk .vtp
    
    return 