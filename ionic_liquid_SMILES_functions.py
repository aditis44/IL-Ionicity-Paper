# Code to manipulate/obtain information from ionic liquid SMILES strings or cation/anion SMILES strings
# This code has functions for determining the net charge of the molecule represented by the input SMILES string, 
# 	determine the cation/anion stoichiometric coefficients to create a neutral IL given the cation/anion SMILES strings, 
#	create the SMILES string for the IL given the cation/anion SMILES strings, obtain the cation/anion SMILES strings given the IL SMILES string
#	calculate the ionic radii for the IL based on its SMILES string using RDKit, Mordred or PubChemPy packages
#	classifying/determining the IL cation family given the cation SMILES string, determining the longest alkyl chain in the IL cation

#Imports
import numpy as np
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.Chem.rdMolDescriptors
import pubchempy as pcp
import mordred
from mordred import VdwVolumeABC, McGowanVolume, MoeType

#SMILES references:
## https://archive.epa.gov/med/med_archive_03/web/html/smiles.html
## https://www.daylight.com/meetings/summerschool98/course/dave/smiles-disco.html
## https://www.daylight.com/dayhtml_tutorials/languages/smiles/index.html

#get_net_charge_smiles takes in a smiles string and returns the net charge of the molecule represented by that string
#if print_level == 1, then additional statements will be printed out as code is run
#NOTE: assumes formatting of SMILES strings where charge is specified by either + or - sign & value of the charge (i.e. +2 vs +4) is denoted by either number of +/- signs or an integer, and all info relating to the charge is within brackets ([]) 
def get_net_charge_smiles(smiles_string, print_level = 0):
	#count number of + & - charges in smiles string
	#Note: could have +2 represented as +2 or ++, but charge will be within brackets []
	#split string around ']'
	charge_list = smiles_string.split(']')
	if(print_level == 1):
		print(f'\nDetermining net charge of input molecule represented by: {smiles_string}')
		print(f'    String split around ]: {charge_list}')
	pos_charge_count = 0
	neg_charge_count = 0
	for smiles_substr in charge_list:
		if('+' in smiles_substr):
			index_pos = smiles_substr.index('+')
			#If + sign is not last char in string, check if the + is followed by a number or another +
			num_characters_after_charge = len(smiles_substr) - index_pos - 1 #subtract 1 since indexing starts at 0
			if(num_characters_after_charge == 0):
				#charge on the atom is just +1
				pos_charge_count += 1
			elif(num_characters_after_charge > 1):
				#Then likely have +++ or ++++, etc. (not +2 or +1)
				num_pos_charges_in_string = smiles_substr.count('+')
				pos_charge_count += num_pos_charges_in_string
			else: #num_characters_after_charge == 1
				#Either have int after '+' indicating the charge or another '+' (indicating a +2 charge)
				if(smiles_substr[index_pos + 1] == '+'):
					#have ++
					pos_charge_count += 2
				else: #have int after '+'
					pos_charge_count += int(smiles_substr[index_pos + 1])
		elif('-' in smiles_substr): #have negative charge
			index_neg = smiles_substr.index('-')
			#If - sign is not last char in string, check if the - is followed by a number or another -
			num_characters_after_charge = len(smiles_substr) - index_neg - 1 #subtract 1 since indexing starts at 0
			if(num_characters_after_charge == 0):
				#charge on the atom is just -1
				neg_charge_count += 1
			elif(num_characters_after_charge > 1):
				#Then likely have --- or ----, etc. (not -2 or -1)
				num_neg_charges_in_string = smiles_substr.count('-')
				neg_charge_count += num_neg_charges_in_string
			else: #num_characters_after_charge == 1
				#Either have int after '-' indicating the charge or another '-' (indicating a -2 charge)
				if(smiles_substr[index_neg + 1] == '-'):
					#have --
					neg_charge_count += 2
				else: #have int after '-'
					neg_charge_count += int(smiles_substr[index_neg + 1])
	net_charge = pos_charge_count - neg_charge_count
	if(print_level == 1):
		print(f'    # of positive charges =  {pos_charge_count}')
		print(f'    # of negative charges =  {neg_charge_count}')
		print(f'Net charge: {net_charge} \n')
	return net_charge



#get_stoichiometric_coefficients_IL takes in the cation SMILES string and anion SMILES string
#by determining the charge on each ion, the stoichiometric coefficients for the cation and anion are returned assuming the IL is neutral
def get_stoichiometric_coefficients_IL(cation_smiles, anion_smiles, print_level = 0):
	cation_charge = get_net_charge_smiles(cation_smiles)
	anion_charge = get_net_charge_smiles(anion_smiles)
	if(print_level == 1):
		if(cation_charge <= 0):
			print(f'Likely error in cation SMILES - charge found to be {cation_charge} <= 0')
		if(anion_charge >= 0):
			print(f'Likely error in anion SMILES - charge found to be {anion_charge} >= 0')
	cation_stoich_coef = None
	anion_stoich_coef = None
	if(np.abs(cation_charge) == np.abs(anion_charge)):
		#magnitude of charges are equal so stoichiometric coefficients both == 1
		cation_stoich_coef = 1
		anion_stoich_coef = 1
	else:
		#magnitude of cation charge != magnitude of anion charge 
		anion_stoich_coef = np.abs(cation_charge)/np.gcd(np.abs(cation_charge), np.abs(anion_charge))
		cation_stoich_coef = np.abs(anion_charge)/np.gcd(np.abs(cation_charge), np.abs(anion_charge))
	if(print_level == 1):
		#print level is high
		print(f'Cation Charge = {cation_charge}, Anion Charge = {anion_charge}')
		print(f'\tStoichiometric Coefficients: cation = {cation_stoich_coef}, anion = {anion_stoich_coef}')
	
	return cation_stoich_coef, anion_stoich_coef
		
# https://numpy.org/doc/stable/reference/generated/numpy.gcd.html


#Create IL SMILES
#get_il_smiles takes in a cation SMILES string, anion SMILES string (it is assumed these SMILES strings are canonicalized SMILES strings)
#returns the SMILES string for the corresponding ionic liquid (assuming neutral charge)
def get_il_smiles(cation_smiles, anion_smiles, print_level = 0):
	cation_stoich_coef, anion_stoich_coef = get_stoichiometric_coefficients_IL(cation_smiles, anion_smiles, print_level)
	curr_il_canonical_smiles = ''
	for j in range(int(cation_stoich_coef)):
		if(j > 0):
			curr_il_canonical_smiles += '.'
		curr_il_canonical_smiles += cation_smiles
	
	for k in range(int(anion_stoich_coef)):
		curr_il_canonical_smiles += '.'
		curr_il_canonical_smiles +=  anion_smiles
	return curr_il_canonical_smiles

#References
#1. https://www.rdkit.org/docs/GettingStartedInPython.html (RDKit documentation - "Note that the SMILES provided is canonical, so the output should be the same no matter how a particular molecule is input")
#2. https://stackoverflow.com/questions/60211666/rdkit-how-to-check-molecules-for-exact-match



#get_cation_anion_smiles takes in a single smiles string that includes both the cation and anion structures
#returns the smiles string of the cation and anion separately
#If print_level = 1, additional statements will be printed out when function runs
def get_cation_anion_smiles(smiles_string, print_level = 0):
	smiles_strings_list = smiles_string.split('.')
	cation = None
	anion = None
	#Separate out cation & anion --> cation has net positive charge, anion has net negative charge
	#get net charge for each smiles string
	for ion in smiles_strings_list:
		net_charge = get_net_charge_smiles(ion, print_level)
		if (net_charge > 0): #cation (net positive charge)
			if(cation is None):
				cation = ion
			elif(ion == cation):
				#if cation is not None but the current ion and cation found already are the same
				if(print_level == 1):
					print('There are multiple of the same cations found in the compound to ensure charge neutrality')
			else:
				sys.exit('Error: multiple different cations present in system, need to modify function')
		elif(net_charge < 0): #anion (net negative charge)
			if(anion is None):
				anion = ion
			elif(ion == anion):
				#if anion is not None but the current ion and anion found already are the same
				if(print_level == 1):
					print('There are multiple of the same anion found in the compound to ensure charge neutrality')
			else:
				sys.exit('Error: multiple different anions present in system')
	if((anion is None) or (cation is None)):
		sys.exit('Error: did not find either an anion or cation')
	else:
		if(print_level == 1):
			print('Successfully identified cation and anion SMILES strings')
		return cation, anion

#Functions below used to estimate ionic radii
#for more analysis on ways to estimate ionic radii, see "estimating_ionic_radii.ipynb"

#Function to calculate radius given volume & surface area and assuming spherical geometry
#get_radius_from_volume_surfaceArea takes in the volume and surface area of a sphere and returns the radius
def get_radius_from_volume_surfaceArea(volume, surfaceArea, print_level = 0):
	radius = 3*volume/surfaceArea
	if(print_level == 1):
		#print out result
		print(f'Resulting radius: {radius}')
	return radius

#get_ionic_radii_McGowanVol_Mordred returns the estimated radius for the molecule calculated using mordred
#uses the McGowan volume descriptor from mordred
#uses mordred.McGowanVolume.McGowanVolume (https://mordred-descriptor.github.io/documentation/master/api/mordred.McGowanVolume.html#mordred.McGowanVolume.McGowanVolume)
def get_ionic_radii_McGowanVol_Mordred(smiles_str, print_level = 0):
	molecule = rdkit.Chem.MolFromSmiles(smiles_str)
	
	mcGowan_volume_calc = mordred.McGowanVolume.McGowanVolume()
	mcGowan_volume = mcGowan_volume_calc(molecule) #believe units outputted by mordred are in Å^3

	#assume volume units are in Å^3
	#Assume spherical geometry --> V = 4/3 pi*r^3, r = (3/4 V/pi)^(1/3)
	r_angstrom = np.cbrt((3/4)*mcGowan_volume/np.pi)
	
	#convert to meters
	r_meters = r_angstrom/(1e10) #10^-10 m = 1 Å
	
	if(print_level == 1):
		#print out result
		print(f'Estimated radius (using mordred McGowanVolume): {r_meters:.5} m')
	
	return r_meters

#get_ionic_radii_dclv_vdWVol_RDKit takes in the SMILES string for a molecule and returns the radius in meters assuming a hard-sphere geometry
#uses RDKit.Chem.rdMolDescriptors.DoubleCubicLatticeVolume class & GetVDWVolume() function
#paper describing double cubic lattice method: https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.540160303
def get_ionic_radii_dclv_vdWVol_RDKit(smiles_str, print_level = 0):
	#first use openbabel to convert SMILES string to xyz 
	molecule = rdkit.Chem.MolFromSmiles(smiles_str)
	#add hydrogens (recommended before generating conformer) & embed molecule/generate 3D conformer
	molecule_with_Hs = rdkit.Chem.AddHs(molecule)
	rdkit.Chem.AllChem.EmbedMolecule(molecule_with_Hs)
	
	#minimize geometry using MMFF94 force field (https://www.rdkit.org/docs/GettingStartedInPython.html)
	rdkit.Chem.AllChem.MMFFOptimizeMolecule(molecule_with_Hs)
	
	#Occassionally, get ValueError: Bad Conformer Id - if so, return None
	try:
		dclv = rdkit.Chem.rdMolDescriptors.DoubleCubicLatticeVolume(molecule_with_Hs, probeRadius=0) #set solvent probe radius to 0 since want van der Waals volume & surface area
	except ValueError as error:
		print(f'Got ValueError: {error}')
		print('Returning None for ionic radii')
		return None
	
	vol = dclv.GetVDWVolume()
	
	#believe volume units are in Å^3 (https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html)
	#Assume spherical geometry --> V = 4/3 pi*r^3, r = (3/4 V/pi)^(1/3)
	r_angstrom = np.cbrt((3/4)*vol/np.pi)
	
	#convert to meters
	r_meters = r_angstrom/(1e10) #10^-10 m = 1 Å
	
	if(print_level == 1):
		#print out result
		print(f'Estimated radius (using RDKit DCLV GetVDWVolume): {r_meters:.5} m')
	
	return r_meters

#get_ionic_radii_computeMolVolume_RDKit takes in the SMILES string for a molecule and returns the radius in meters assuming a hard-sphere geometry
#uses RDKit.Chem.AllChem.ComputeMolVolume
def get_ionic_radii_computeMolVolume_RDKit(smiles_str, print_level = 0):
	#first use openbabel to convert SMILES string to xyz 
	molecule = rdkit.Chem.MolFromSmiles(smiles_str)
	#add hydrogens (recommended before generating conformer) & embed molecule/generate 3D conformer
	molecule_with_Hs = rdkit.Chem.AddHs(molecule)
	rdkit.Chem.AllChem.EmbedMolecule(molecule_with_Hs)
	
	#minimize geometry using MMFF94 force field (https://www.rdkit.org/docs/GettingStartedInPython.html)
	rdkit.Chem.AllChem.MMFFOptimizeMolecule(molecule_with_Hs)
	
	#Occassionally, get ValueError: Bad Conformer Id - if so, return None
	try:
		vol = rdkit.Chem.AllChem.ComputeMolVolume(molecule_with_Hs)
	except ValueError as error:
		print(f'Got ValueError: {error}')
		print('Returning None for ionic radii')
		return None

	#believe volume units are in Å^3 (https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html)
	#Assume spherical geometry --> V = 4/3 pi*r^3, r = (3/4 V/pi)^(1/3)
	r_angstrom = np.cbrt((3/4)*vol/np.pi)
	
	#convert to meters
	r_meters = r_angstrom/(1e10) #10^-10 m = 1 Å
	
	if(print_level == 1):
		#print out result
		print(f'Estimated radius (using RDKit ComputeMolVolume): {r_meters:.5} m')
	
	return r_meters

#get_ionic_radii_PubChemPy takes in the SMILES string for a molecule and returns the radius in meters assuming a hard-sphere geometry
#uses PubChemPy.get_properties
def get_ionic_radii_PubChemPy(smiles_str, print_level = 0):
	#Occasionally, get Key Error PubChemPy when obtaining Volume3D
	#https://docs.python.org/3/tutorial/errors.html, https://www.geeksforgeeks.org/how-to-handle-keyerror-exception-in-python/
	try:
		vol = pcp.get_properties(properties = 'Volume3D',identifier=smiles_str, namespace='smiles')[0]['Volume3D']
	except KeyError as error:
		print(f'Got KeyError: {error}')
		print('Returning None for ionic radii')
		return None
		
	vol = pcp.get_properties(properties = 'Volume3D',identifier=smiles_str, namespace='smiles')[0]['Volume3D']
	#assume volume units are in Å^3
	#Assume spherical geometry --> V = 4/3 pi*r^3, r = (3/4 V/pi)^(1/3)
	r_angstrom = np.cbrt((3/4)*vol/np.pi)
	
	#convert to meters
	r_meters = r_angstrom/(1e10) #10^-10 m = 1 Å
	
	if(print_level == 1):
		#print out result
		print(f'Estimated radius (using PubChemPy): {r_meters:.5} m')
	
	return r_meters

#get_ionic_radii_dclv_vdWVol_SA_RDKit takes in the SMILES string for a molecule and returns the radius in meters assuming a hard-sphere geometry
#uses RDKit.Chem.rdMolDescriptors.DoubleCubicLatticeVolume class, GetVDWVolume(), and GetSurfaceArea() functions
#paper describing double cubic lattice method: https://onlinelibrary.wiley.com/doi/abs/10.1002/jcc.540160303
def get_ionic_radii_dclv_vdWVol_SA_RDKit(smiles_str, print_level = 0):
	#first use openbabel to convert SMILES string to xyz 
	molecule = rdkit.Chem.MolFromSmiles(smiles_str)
	#add hydrogens (recommended before generating conformer) & embed molecule/generate 3D conformer
	molecule_with_Hs = rdkit.Chem.AddHs(molecule)
	rdkit.Chem.AllChem.EmbedMolecule(molecule_with_Hs)
	
	#minimize geometry using MMFF94 force field (https://www.rdkit.org/docs/GettingStartedInPython.html)
	rdkit.Chem.AllChem.MMFFOptimizeMolecule(molecule_with_Hs)
	
	#Occassionally, get ValueError: Bad Conformer Id - if so, return None
	try:
		dclv = rdkit.Chem.rdMolDescriptors.DoubleCubicLatticeVolume(molecule_with_Hs, probeRadius=0) #set solvent probe radius to 0 since want van der Waals volume & surface area
	except ValueError as error:
		print(f'Got ValueError: {error}')
		print('Returning None for ionic radii')
		return None

	vol_angstromsCubed = dclv.GetVDWVolume() #believe units are in Å^3
	vol_metersCubed = vol_angstromsCubed*((1e-10)**3) #convert to m^3; 10^-10 m = 1Å
	
	surface_area_angstromsSquared = dclv.GetSurfaceArea() #believe units are in Å^2
	surface_area_metersSquared = surface_area_angstromsSquared*((1e-10)**2) #convert to m^2; 10^-10 m = 1Å
	
	r_meters = get_radius_from_volume_surfaceArea(vol_metersCubed, surface_area_metersSquared)

	if(print_level == 1):
		#print out result
		print(f'Estimated Van der Waals Volume (RDKit DCLV GetVDWVolume): {vol_metersCubed} m^3, radius using volume: {np.cbrt((3/4)*vol_metersCubed/np.pi)} m')
		print(f'Estimated Surface Area (RDKit DCLV GetSurfaceArea): {surface_area_metersSquared} m^2, radius using SA: {np.sqrt((1/4)*surface_area_metersSquared/np.pi)} m')
		print(f'Estimated radius (using RDKit DCLV GetVDWVolume & GetSurfaceArea): {r_meters:.5} m')
	
	return r_meters

#Functions below used to classify/determine IL cation family & longest alkyl chain in the IL

#create_default_cation_family_list creates a default list of RDKit molecules corresponding to different cation families
#returns a list of rdkit molecules for each cation family & the associated cation family name (default includes imidazolium, pyridinium, pyrazolium, pyrrolidinium, piperidinium, morpholinium, triazolium, ammonium, phosphonium, and sulfonium)
def create_default_cation_family_list():
	#Generated SMILES strings using https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html
	cation_family_list = []
	cation_family_names = []

	imidazolium_smiles = 'c1[n+](ccn1-[*])-[*]'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(imidazolium_smiles))
	cation_family_names.append('imidazolium')

	pyridinium_smiles = '*[N+]1(*)C=CC=CC1'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(pyridinium_smiles))
	cation_family_names.append('pyridinium')

	pyridinium_smiles_2 = '*[n+]1ccccc1' #another representation of pyridinium cations: https://www.researchgate.net/figure/Molecular-structures-of-the-pyridinium-based-organic-cations-used-in-this-study_fig1_339393616
	cation_family_list.append(rdkit.Chem.MolFromSmiles(pyridinium_smiles_2))
	cation_family_names.append('pyridinium')

	pyrazolium_smiles = '*n1ccc[n+]1*'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(pyrazolium_smiles))
	cation_family_names.append('pyrazolium')

	pyrrolidinium_smiles = '*[N+]1(*)CCCC1'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(pyrrolidinium_smiles))
	cation_family_names.append('pyrrolidinium')

	piperidinium_smiles = '*[N+]1(*)CCCCC1'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(piperidinium_smiles))
	cation_family_names.append('piperidinium')

	morpholinium_smiles = '*[N+]1(*)CCOCC1'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(morpholinium_smiles))
	cation_family_names.append('morpholinium')

	triazolium_smiles = '*[N+]1(*)C=CN=N1' #based on 1,2,3 triazole
	cation_family_list.append(rdkit.Chem.MolFromSmiles(triazolium_smiles))
	cation_family_names.append('triazolium')

	triazolium_smiles2 = '*n1c[n+](*)cn1' #Based on 1,2,4 triazole
	cation_family_list.append(rdkit.Chem.MolFromSmiles(triazolium_smiles2))
	cation_family_names.append('triazolium')

	triazolium_smiles3 = '*n1cn[n+](*)c1' #Based on 1,2,4 triazole in ILThermo dataset
	cation_family_list.append(rdkit.Chem.MolFromSmiles(triazolium_smiles3))
	cation_family_names.append('triazolium')

	ammonium_smiles = '*[N+](*)(*)*'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(ammonium_smiles))
	cation_family_names.append('ammonium')

	phosphonium_smiles = '*[P+](*)(*)*'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(phosphonium_smiles))
	cation_family_names.append('phosphonium')

	sulfonium_smiles = '*[S+](*)*'
	cation_family_list.append(rdkit.Chem.MolFromSmiles(sulfonium_smiles))
	cation_family_names.append('sulfonium')
	
	return cation_family_list, cation_family_names

#get_cation_family_substructure_search takes in a list of rdkit molecule of the cation(s), a list of the IL compound names associated with each cation in the list,
#optionally takes in a list of rdkit molecules for IL cation families to consider and a list of the names for each family (see create_default_cation_family_list)
#and searches for a substructure match among the cation families in cation_family_list
#returns a list of family names each cation belongs to
#if print_level == 1, additional info is printed as code runs
def get_cation_family_substructure_search(cation_mols, unique_ils_cmpd_names, print_level = 0, cation_family_list=None, cation_family_names=None):
	if(cation_family_list is None):
		if(print_level == 1):
			print('No cation family list provided, will use default (default includes imidazolium, pyridinium, pyrazolium, pyrrolidinium, piperidinium, morpholinium, triazolium, ammonium, phosphonium, and sulfonium)')
		cation_family_list, cation_family_names = create_default_cation_family_list()

	il_names = {} #dictionary where keys are index of molecule in cation_moleculecs and entry is the cation family name
	for j in range(len(cation_family_list)):
		curr_cation_family  = cation_family_list[j]
		curr_cation_family_name = cation_family_names[j]
		results, ils_not_classified_index = rdkit.Chem.rdRGroupDecomposition.RGroupDecompose(cores = curr_cation_family, mols = cation_mols)
		for i in range(len(cation_mols)):
			if (not(i in ils_not_classified_index)): #this IL belongs to curr_cation_family
				if(il_names.get(i) is None): #haven't yet assigned a family to this IL
					if(print_level == 1): #print additional info
						print(f'\n****** Classified IL {unique_ils_cmpd_names[i]} as belonging to {curr_cation_family_name}')
					il_names[i] = curr_cation_family_name #so assign a family to it
				else:
					#Since cations belonging to imidazolium, pyridinium, pyrazolium, pyrrolidinium, piperidinium, morpholinium, and triazolium could also be classified as belonging to ammonium family, choose former cation family for it to belong to
					#However, if ammonium is not the other cation family the IL has been classified as belonging to, throw an error
					if(not(curr_cation_family_name == 'ammonium')):
						print(f'\n****** Identified multiple cation families for: {unique_ils_cmpd_names[i]} ******\n')
						print(f'So far these families include: {il_names[i]}, {curr_cation_family_name}')
						sys.exit(f'Multiple cation families identified for {unique_ils_cmpd_names[i]}')
			#else:
				#this IL doesn't belong to current cation family

	il_names_list = []
	ils_not_classified_count = 0
	for i in range(len(cation_mols)):
		if(il_names.get(i) is None): #haven't yet assigned a family to this IL
			il_names_list.append('other')
			ils_not_classified_count+=1
			print(f'\n****** Did not identify cation family for cation: {unique_ils_cmpd_names[i]}, will classify as \'other\' ******\n')
		else:
			il_names_list.append(il_names[i])
	print(f'Out of {len(il_names_list)} ILs, classified {len(il_names_list) - ils_not_classified_count} (had {ils_not_classified_count} ILs classified as \'other\')')
	return il_names_list
#References:
#https://greglandrum.github.io/rdkit-blog/posts/2023-01-09-rgd-tutorial.html#the-basics
#https://rdkit.org/docs/source/rdkit.Chem.rdRGroupDecomposition.html

#create_alkyl_chain_list takes in a maximum alkyl chain length (max # of Cs) for the longest alkyl chain
#returns a list of rdkit molecules w/alkyl chains up to the maximum number provided & a list w/the number of carbons in each chain created
def create_alkyl_chain_list(max_alkyl_chain_length):
	alkyl_chain_smiles_list = []
	alkyl_chain_list = []
	alkyl_chain_num_carbons = []
	alkyl_chain_smiles_maxC = ''
	for i in range(max_alkyl_chain_length):
		alkyl_chain_smiles_maxC += 'C'

	for j in range(max_alkyl_chain_length):
		alkyl_chain_smiles_list.append(alkyl_chain_smiles_maxC[:max_alkyl_chain_length-j])
		#alkyl_chain_smiles_list[j] +='*'
		alkyl_chain_num_carbons.append(str(max_alkyl_chain_length-j))

	for smiles_str in alkyl_chain_smiles_list:
		alkyl_mol = rdkit.Chem.MolFromSmiles(smiles_str)
	   #alkyl_mol_withHs = rdkit.Chem.AddHs(alkyl_mol)
		alkyl_chain_list.append(alkyl_mol) 
	return alkyl_chain_list, alkyl_chain_num_carbons
				  
#get_longest_alkyl_chain_substructure_search takes in an rdkit molecule of the cation, a list of the IL compound names associated with each cation in the list and the maximum length of an alkyl chain to consider (default = 16)
#and searches for a substructure match among the alkyl chains (decreasing in # of carbons) in alkyl_chain_list
#returns a list with longest alkyl chain in each cation
#if print_level == 1, additional info is printed as code runs
def get_longest_alkyl_chain_substructure_search(cation_mols, unique_ils_cmpd_names, max_alkyl_chain_length=16, print_level = 0):
	alkyl_chain_list, alkyl_chain_num_carbons = create_alkyl_chain_list(max_alkyl_chain_length)
	il_names = {} #dictionary where keys are index of molecule in cation_moleculecs and entry is the number of carbons in longest alkyl chain
	for j in range(len(alkyl_chain_list)):
		curr_longest_alkyl_chain  = alkyl_chain_list[j]
		curr_longest_alkyl_chain_name = alkyl_chain_num_carbons[j]
		results, ils_not_classified_index = rdkit.Chem.rdRGroupDecomposition.RGroupDecompose(cores = curr_longest_alkyl_chain, mols = cation_mols)
		for i in range(len(cation_mols)):
			if (not(i in ils_not_classified_index)): #this IL belongs to curr_longest_alkyl_chain
				if(il_names.get(i) is None): #haven't yet assigned a longest alkyl chain to this IL
					#Check that the core match is not a ring with the same number of carbons
					#Check if atoms that match the alkyl chain are in a ring or not
					atom_indices_match_chain_list = cation_mols[i].GetSubstructMatches(curr_longest_alkyl_chain)
					#print(atom_indices_match_chain_list)
					matches_are_rings = []
					for j in range(len(atom_indices_match_chain_list)):
						atom_indices = atom_indices_match_chain_list[j]
						match_is_ring = False
						for k in range(len(atom_indices) - 1):
							ringInfo = cation_mols[i].GetRingInfo()
							if(ringInfo.AreAtomsInSameRing(atom_indices[k], atom_indices[k+1])):
								match_is_ring = True
						if(match_is_ring):
							matches_are_rings.append(True)
						else:
							matches_are_rings.append(False)
					print(matches_are_rings)
					if(False in matches_are_rings):
						if(print_level == 1): #print additional info
							print(f'\n****** Longest alkyl chain in IL {unique_ils_cmpd_names[i]} is: {curr_longest_alkyl_chain_name}')
						il_names[i] = curr_longest_alkyl_chain_name #so assign a longest alkyl chain to it
					else:
						print(f'****** Match found was a ring (not alkyl chain) for: {unique_ils_cmpd_names[i]} ******\n')
			#else:
				#this IL doesn't have alkyl chain w/current length

	il_names_list = []
	ils_not_classified_count = 0
	for i in range(len(cation_mols)):
		if(il_names.get(i) is None): #haven't yet assigned longest alkyl chain length to this IL
			il_names_list.append('0') #After visualizing these ILs, these ILs have no alkyl chains (so say longest chain is 0)
			ils_not_classified_count+=1
			print(f'\n****** Did not identify longest alkyl chain for cation: {unique_ils_cmpd_names[i]}, will classify as \'0\' ******\n')
		else:
			il_names_list.append(il_names[i])
	print(f'Out of {len(il_names_list)} ILs, classified {len(il_names_list) - ils_not_classified_count} (had {ils_not_classified_count} ILs classified as \'0\')')
	return il_names_list
#References:
#https://greglandrum.github.io/rdkit-blog/posts/2023-01-09-rgd-tutorial.html#the-basics
#https://rdkit.org/docs/source/rdkit.Chem.rdRGroupDecomposition.html
