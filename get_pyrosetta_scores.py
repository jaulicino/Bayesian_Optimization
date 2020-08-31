#let's initialize pyrosetta
from pyrosetta import *
from pyrosetta.toolbox import *
from pyrosetta.toolbox.mutants import *
from pyrosetta import PyMOLMover

#and now the rest we will need
import numpy as np
import math
import gc
import utils
import sys
import json
import time

init("-mute all")
pose = Pose()
pose = pose_from_pdb("data/2vkn.clean.pdb")
scorefxn = get_fa_scorefxn()

debug = False

def get_pyrosetta_scores_from_sequence(sequence):
	start_align = 1
	bottom = pose.sequence()[0:63]
	score = 0
	top = pose.sequence()[0:start_align] + sequence
	mutated_pose = Pose()
	mutated_pose.assign(pose)

	##when you delete/insert i is no longer the correct place to insert/delete, must add offset
	offset = 0

	if(len(top) != len(bottom)):
		raise Exception('Alignments should be the same length... t/b length -->' + str(len(top)) + "    " + str(len(bottom)))
	for i in range(len(top)):
		if top[i] == bottom[i]:
			if(debug): print("No change at " + str(i))

		elif (top[i] != "-" and bottom[i] != "-"):
			if(debug): print("*************Mutating    " + str(i)) 
			mutate_residue(mutated_pose, i + 1 + offset, top[i], pack_scorefxn = scorefxn)

		elif top[i] == "-":
			if(debug): print("-------------"+str(i)+"-------------")
			#delete
			utils.delete_residue(mutated_pose,i + 1 + offset)
			offset -= 1

		elif bottom[i] == "-" and i + offset < mutated_pose.total_residue():
			#insert
			print("ERROR SHOULD NOT HAVE - in bottom for align_2")

	score = scorefxn(mutated_pose)
	return score