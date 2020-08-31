from pyrosetta import *
from rosetta.protocols.loops.loop_closure.kinematic_closure import *
import numpy as np
from rosetta.protocols.loops.loop_closure.kinematic_closure import *



init("-mute all")

def delete_residue(pose, number):
	pose.delete_residue_slow(number)
	if(number <= 2):
		return
	if(number >= pose.total_residue()-1):
		return
	kic_mover = KinematicMover()
	kic_mover.set_pivots(number-1, number, number+1)
	kic_mover.apply(pose)


def insert_residue(pose, number, letter):
	chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
	resiset = chm.residue_type_set( 'fa_standard' )
	res_type = resiset.get_representative_type_name1(letter) #e.g. A
	residue = pyrosetta.rosetta.core.conformation.ResidueFactory.create_residue(res_type)

	if(number > 0):
		pose.append_polymer_residue_after_seqpos(residue, number, True)
	elif(number>=0):
		pose.prepend_polymer_residue_before_seqpos(residue, number+1, True)
	else:
		raise "ERROR I + Offset should be positive in insertion"


