#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include <ale_interface.hpp>
#include "population.h"

class Program
{
public:
	Program();
	void Run();

private:
	// Meny functions
	void Config();
	void Info() const;
	void LoadAgent();
	void LoadRom();
	void Play();
	std::vector<float> ProcessInput(const std::vector<unsigned char> &input) const;
	void Train();

	// CONFIG
	unsigned int ALE_RANDOM_SEED;
	unsigned int FACTOR_DOWNSCALED_X;
	unsigned int FACTOR_DOWNSCALED_Y;
	unsigned int FACTOR_DOWNSCALED_MULTIPLE;
	unsigned int GENERATIONS;
	unsigned int MAXIMUM_NUMBER_OF_FRAMES;
	unsigned int PIXELS_X;
	unsigned int PIXELS_Y;
	unsigned int PIXELS_DOWNSCALED_X;
	unsigned int PIXELS_DOWNSCALED_Y;
	float DOWNSCALE_FACTOR;
	unsigned int SENSOR_INPUTS;

	// Variables
	bool mIsRomLoaded;
	ale::ALEInterface mALE;
	NEAT::Organism *mAgent;
};

#endif
