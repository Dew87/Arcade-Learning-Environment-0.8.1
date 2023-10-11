#ifndef _FNN_H_
#define _FNN_H_

namespace FNN
{
	extern int elitism;
	extern int hidden_nodes;
	extern int pop_size;
	extern double mate_multipoint_prob;
	extern double mate_multipoint_avg_prob;
	extern double mate_singlepoint_prob;
	extern double mutate_link_weights_cap;
	extern double mutate_link_weights_prob;
	extern double mutate_link_weights_ratio_gaussian_to_coldgaussian;
	extern double weight_mut_power;  // The power of a linkweight mutation 

	bool load_fnn_params(const char *filename, bool output = false);
	void print_fnn_params();
}

#endif
