# Simulations using OpenFOAM

## Preparing modules on Euler
Courtesy of Jonathan Nestmann
### Usage of OpenFOAM module
The Euler HPC offers a variety of preinstalled modules to use for computation. These modules have to be loaded in the shell. For the usage of the precompiled OpenFOAM (v1806) of Euler insert the following commands in the *~/.bash_profile* file.

```
module load gcc/4.8.5 openmpi/3.0.1
module load openfoam-com/1806
module load python/3.6.4
``` 

### Additional package installation 
For certain haemodynamic applications additional packages of OpenFOAM are required. Besides using an isolated container, it is also possible to compile packages in your home directory directly. Before installing new packages for OpenFOAM, always make sure the required modules of OpenFOAM are loaded.

#### Compilation of swak4Foam
This packages includes the groovy boundary condition and multiple other useful functions. Download the repository onto your local machine as described in [swak4Foam](https://openfoamwiki.net/index.php/Contrib/swak4Foam). Make sure to use the Mercurial repository, therefore install Mercurial onto your local machine first via [Mercurial](https://wiki.mercurial-scm.org/Download). \\
Pull the repository via:
```
hg clone http://hg.code.sf.net/p/openfoam-extend/swak4Foam swak4Foam
```
Copy the repository to the Euler cluster, e.g. via sshsf-mounting a folder or copying by rsync:
```
rsync -a PATH/swak4Foam <eth-login>@euler.ethz.ch:/cluster/home/<eth-login>
```


For a full compilation of swak4Foam, some OpenFOAM variables have to exported first. Run the following lines in Euler after loading OpenFOAM (see above):
```
mkdir $HOME/.OpenFOAM
echo 'export WM_PROJECT_USER_DIR=$HOME/$WM_PROJECT/$USER-$WM_PROJECT_VERSION' > $HOME/.OpenFOAM/prefs.sh
echo 'export FOAM_USER_APPBIN=$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin' >> $HOME/.OpenFOAM/prefs.sh
echo 'export FOAM_USER_LIBBIN=$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib'  >> $HOME/.OpenFOAM/prefs.sh
echo 'export PATH=$PATH:$FOAM_USER_APPBIN'  >> $HOME/.OpenFOAM/prefs.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$FOAM_USER_LIBBIN'  >> $HOME/.OpenFOAM/prefs.sh
source $HOME/.OpenFOAM/prefs.sh
```

Add the following line after the loading commands of OpenFOAM in the *~/.bash_profile*:
```
source $HOME/.OpenFOAM/prefs.sh
```

The **Installation** compiles the package with all the OpenFOAM variables included:
```
cd PATH/swak4Foam
./AllwmakeAll
```

For full usability add the following variable to the *~/.bash_profile* file:
```
SWAK4FOAM_SRC="/cluster/home/<eth-login>/<PATH>/swak4Foam/Libraries"
```


#### Compilation of haemoFoam
The package [haemoFoam](https://github.com/TS-CUBED/haemoFoam) allows to use specific functionalities for the simulation of haemodynamics, e.g. non-Newtonian properties or windkessel boundary condition. Make sure to compile swak4Foam before installing the haemoFoam package. Pull the repository and compile the package:
```
git clone https://github.com/TS-CUBED/haemoFoam.git
cd /haemoFoam
./Allwmake
```


### Compilation of user-specific OpenFOAM version
To build a OpenFOAM from source, activate the following packages in the shell. 
```
module load gcc/8.2.0
module load openmpi/4.1.4
module load scotch/6.0.7
module load cmake/3.25.0
```
Clone the desired OpenFOAM repository to Euler:
```
cd $HOME/OpenFOAM
git clone https://github.com/OpenFOAM/OpenFOAM-10.git
```
Compile OpenFOAM:
```
cd OpenFOAM-10
./Allwmake
echo "source $HOME/OpenFOAM/OpenFOAM-10/etc/bashrc" >> ~/.bashrc
source etc/bashrc
```
Install ThirdParty tools:
```
git clone https://github.com/OpenFOAM/ThirdParty-10.git
mv ThirdParty-10 $WM_THIRD_PARTY_DIR
module load gcc/8.2.0 openmpi/4.1.4
./Allwmake > log.Allwmake 2>&1
```