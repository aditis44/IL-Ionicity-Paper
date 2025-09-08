#Code to create structure input file for the conformer generation pipeline for openCOSMO-RS package
#Link to GitHub code for openCOSMO-RS conformer pipeline: https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline
#OpenCOSMO-RS paper: Gerlach, T., Müller, S., de Castilla, A. G., & Smirnova, I. (2022). An open source COSMO-RS implementation and parameterization supporting the efficient implementation of multiple segment descriptors. Fluid phase equilibria, 560, 113472.

#Imports
import argparse

#Read in values of parameters to include in structure input file
#Can pass 6 arguments to this code: path_to_save_input_file, name of the structure, SMILES string, xyz file (optional), charge, and whether to do a geometry optimization
#The structure name will also be the name used in saving the structure file (i.e. if name is ethanol, input file will be saved as conformer_gen_input_ethanol.inp)
parser = argparse.ArgumentParser(description="input parameters for structure input file for use with openCOSMO-RS conformer pipeline")
parser.add_argument('-path_to_save_input_file', type=str, dest='path_to_save_input_file', help='Path to save input file to')
parser.add_argument('-structure_name', type=str, dest='structure_name', help='Name of structure, will be included in structure input file & in filename (filename will be conformer_gen_input_[structure_name].inp)')
parser.add_argument('-smiles_str', type=str, dest='smiles_str', help='SMILES string for input structure')
parser.add_argument('-xyz_filename', type=str, dest='xyz_filename', help='xyz filename (if do not want to provide an xyz file, input should be 0, everything else will be treated as a path to an xyz file)')
parser.add_argument('-charge', type=int, dest='charge', help='Charge on input structure')
parser.add_argument('-geometry_optimization', type=str, dest='geometry_optimization', help='Whether or not to do geometry optimization (if want to do geometry optimization, input should be True, everything else will be treated as False)')

args = parser.parse_args()
path_to_save_input_file = args.path_to_save_input_file
structure_name = args.structure_name
smiles_str = args.smiles_str
xyz_filename = args.xyz_filename
if(xyz_filename == '0'):
    #do not want to provide xyz file as input
    xyz_input_str = ''
else:
    xyz_input_str = xyz_filename

charge = args.charge
geometry_optimization = args.geometry_optimization
if(geometry_optimization == 'True'):
    #will indicate in input file to do geometry optimization
    geometry_optimization_input_str = 'True'
else:
    geometry_optimization_input_str = 'False'

structure_input_filename = path_to_save_input_file + f'/conformer_gen_input_{structure_name}.inp'
structure_file = open(structure_input_filename, "w")
#write the file following the format specified in README for conformer generation code (with file.inp beeing a TAB separated file similar to the following: name [TAB] SMILES [TAB] optional xyz file [TAB] charge [TAB] optional geometry optimization (the optional geometry optimization column assumes [0, False, no] to be False and everything else to be True))
structure_file.write(f'{structure_name}\t{smiles_str}\t{xyz_input_str}\t{charge}\t{geometry_optimization_input_str}')
structure_file.close()
