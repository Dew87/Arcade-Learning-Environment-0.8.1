#include "FNN.h"
#include <fstream>
#include <iostream>

using namespace std;

int FNN::elitism = 0;
int FNN::hidden_nodes = 0;
int FNN::pop_size = 0;
double FNN::mate_multipoint_prob = 0;
double FNN::mate_multipoint_avg_prob = 0;
double FNN::mate_singlepoint_prob = 0;
double FNN::mutate_link_weights_cap = 0;
double FNN::mutate_link_weights_prob = 0;
double FNN::mutate_link_weights_ratio_gaussian_to_coldgaussian = 0;
double FNN::weight_mut_power = 0; // The power of a linkweight mutation 

bool FNN::load_fnn_params(const char *filename, bool output)
{
	ifstream ifs(filename);

	if (!ifs)
	{
		return false;
	}

	string temp;

	// **********LOAD IN PARAMETERS*************** //
	if (output)
	{
		cout << "FNN READING IN " << filename << "\n";
	}

	ifs >> temp >> FNN::elitism;
	ifs >> temp >> FNN::hidden_nodes;
	ifs >> temp >> FNN::pop_size;
	ifs >> temp >> FNN::mate_multipoint_prob;
	ifs >> temp >> FNN::mate_multipoint_avg_prob;
	ifs >> temp >> FNN::mate_singlepoint_prob;
	ifs >> temp >> FNN::mutate_link_weights_cap;
	ifs >> temp >> FNN::mutate_link_weights_prob;
	ifs >> temp >> FNN::mutate_link_weights_ratio_gaussian_to_coldgaussian;
	ifs >> temp >> FNN::weight_mut_power;

	if (output)
	{
		print_fnn_params();
	}

	ifs.close();
	return true;
}

void FNN::print_fnn_params()
{
	cout << "elitism = " << FNN::elitism << "\n";
	cout << "hidden_nodes = " << FNN::hidden_nodes << "\n";
	cout << "pop_size = " << FNN::pop_size << "\n";
	cout << "mate_multipoint_prob = " << FNN::mate_multipoint_prob << "\n";
	cout << "mate_multipoint_avg_prob = " << FNN::mate_multipoint_avg_prob << "\n";
	cout << "mate_singlepoint_prob = " << FNN::mate_singlepoint_prob << "\n";
	cout << "mutate_link_weights_cap  = " << FNN::mutate_link_weights_cap << "\n";
	cout << "mutate_link_weights_prob = " << FNN::mutate_link_weights_prob << "\n";
	cout << "mutate_link_weights_ratio_gaussian_to_coldgaussian  = " << FNN::mutate_link_weights_ratio_gaussian_to_coldgaussian << "\n";
	cout << "weight_mut_power = " << FNN::weight_mut_power << "\n";
}
