#ifndef _FNNPOPULATION_H_
#define _FNNPOPULATION_H_

#include <vector>
#include "NEAT/organism.h"

namespace FNN
{
	class FNNPopulation
	{
	public:
		FNNPopulation(int num_in, int num_out, int num_hidden, int size);
		~FNNPopulation();

		// Turnover the population to a new generation using fitness 
		// The generation argument is the next generation
		bool epoch(int generation);

		// Run verify on all Genomes in this Population (Debugging)
		bool verify();

		std::vector<NEAT::Organism*> organisms; // The organisms in the Population

		// ******* When do we need to delta code? *******
		double highest_fitness;  // Stagnation detector
		int highest_last_changed;

	private:
		// Delete population
		void erase();

		// Selection Roulette Wheel
		NEAT::Organism* SelectionRouletteWheel(double total_fitness) const;
	};
}

#endif
