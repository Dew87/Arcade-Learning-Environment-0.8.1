#include "FNN.h"
#include "FNNPopulation.h"
#include "NEAT/organism.h"
#include <iostream>
#include <sstream>
#include <fstream>

using namespace FNN;
using namespace std;

FNNPopulation::FNNPopulation(int num_in, int num_out, int num_hidden, int size)
{
	highest_fitness = 0.0;
	highest_last_changed = 0;

	for (int i = 1; i <= size; i++)
	{
		NEAT::Genome *new_genome = new NEAT::Genome(num_in, num_out, num_hidden, 3);
		new_genome->genome_id = i;
		NEAT::Organism *new_organism = new NEAT::Organism(0.0, new_genome, 1);
		organisms.push_back(new_organism);
	}
}

FNNPopulation::~FNNPopulation()
{
	erase();
}

bool FNNPopulation::epoch(int generation)
{
	vector<NEAT::Organism*> next_generation;

	vector<NEAT::Organism*>::iterator curorg;

	NEAT::Organism* champion;

	double total = 0.0; // Used to compute average fitness over all Organisms
	double overall_average = 0.0; // The average modified fitness among ALL organisms

	sort(organisms.begin(), organisms.end(), NEAT::order_orgs);
	champion = *organisms.begin();

	// Go through the organisms and add up their fitnesses to compute the overall average
	for (vector<NEAT::Organism*>::iterator i = organisms.begin(); i != organisms.end(); ++i)
	{
		total += (*i)->fitness;
	}
	overall_average = total / organisms.size();
	cout << "Generation " << generation << ": fitness average " << overall_average << " max " << champion->fitness << "\n";

	// Check if new population highest fitness
	if (champion->fitness > highest_fitness)
	{
		highest_fitness = champion->fitness;
		highest_last_changed = 0;
		cout << "NEW POPULATION RECORD FITNESS: " << highest_fitness << endl;
	}
	else
	{
		++highest_last_changed;
		cout << highest_last_changed << " generations since last population fitness record: " << highest_fitness << endl;
	}

	// Spawn next generation
	for (int count = 1; count <= organisms.size(); ++count)
	{
		NEAT::Organism *mom; // Parent organisms
		NEAT::Organism *dad; // Parent organisms
		NEAT::Organism *baby; // The new organism

		NEAT::Genome *new_genome; // For holding baby's genes

		if (count <= FNN::elitism) // Elitism
		{
			mom = champion;
			new_genome = (mom->gnome)->duplicate(count);

			// Mutate babies except the first
			if (1 < count)
			{
				new_genome->mutate_link_weights(FNN::weight_mut_power, 1.0, NEAT::GAUSSIAN);
			}
		}
		else
		{
			// Selection
			mom = SelectionRouletteWheel(total);
			dad = SelectionRouletteWheel(total);

			// Mating
			if (NEAT::randfloat() < FNN::mate_multipoint_prob)
			{
				new_genome = (mom->gnome)->mate_multipoint(dad->gnome, count, mom->orig_fitness, dad->orig_fitness, false);
			}
			else if (NEAT::randfloat() < (FNN::mate_multipoint_avg_prob / (FNN::mate_multipoint_avg_prob + FNN::mate_singlepoint_prob)))
			{
				new_genome = (mom->gnome)->mate_multipoint_avg(dad->gnome, count, mom->orig_fitness, dad->orig_fitness, false);
			}
			else
			{
				new_genome = (mom->gnome)->mate_singlepoint(dad->gnome, count);
			}

			// Mutation
			if (NEAT::randfloat() < FNN::mutate_link_weights_prob)
			{
				new_genome->mutate_link_weights(FNN::weight_mut_power, 1.0, NEAT::GAUSSIAN);
			}
		}

		// Create baby and add to next generation
		baby = new NEAT::Organism(0.0, new_genome, generation);
		next_generation.push_back(baby);
	}

	// Delete old generation
	erase();

	// Set next generation as current population
	organisms = next_generation;

	return true;
}

void FNNPopulation::erase()
{
	for (vector<NEAT::Organism*>::iterator i = organisms.begin(); i != organisms.end(); ++i)
	{
		delete *i;
	}
	organisms.clear();
}

NEAT::Organism* FNNPopulation::SelectionRouletteWheel(double total_fitness) const
{
	// If total fitness == 0. Random individual
	if (total_fitness == 0.0)
	{
		int i = NEAT::randint(0, organisms.size() - 1);
		return organisms[i];
	}
	else
	{
		double marble = NEAT::randfloat() * total_fitness;
		vector<NEAT::Organism*>::const_iterator curorg = organisms.begin();
		double spin = (*curorg)->fitness;
		while (spin < marble && curorg != organisms.end())
		{
			++curorg;
			spin += (*curorg)->fitness;
		}
		return (*curorg);
	}
}

bool FNNPopulation::verify()
{
	vector<NEAT::Organism*>::iterator curorg;

	bool verification = true;

	for (curorg = organisms.begin(); curorg != organisms.end(); ++curorg)
	{
		verification = ((*curorg)->gnome)->verify();
	}

	return verification;
}
