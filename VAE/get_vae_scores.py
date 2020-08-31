from vae_score import *



def get_vae_scores_from_fasta(fasta_file):
	alignment = fasta_file
	sequence = [str(i) for i in toolkit.get_seq(alignment)]
	sequence = [i[:16]+i[18:44]+i[45:] for i in toolkit.get_seq(alignment)]
	v_traj_onehot, _ = toolkit.convert_potts(sequence, index)
	log_p_list = np.array(toolkit.make_logP(v_traj_onehot, p_weight,q_n))
	return log_p_list

print(get_scores("test.fasta"))
