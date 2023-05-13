#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include <ale_interface.hpp>
#include <NEAT/organism.h>

class Program
{
public:
	Program();
	void Run();

private:
	// Menu functions
	void Config();
	void LoadAgent();
	void LoadRom();
	void LogStart(std::ofstream &log);
	void Play(size_t games, bool SDL);
	void Print() const;
	std::vector<float> ProcessInput(const std::vector<unsigned char> &input) const;
	void TrainFNN();
	void TrainNEAT();

	// CONFIG
	unsigned int ALE_RANDOM_SEED;
	float ALE_REPEAT_ACTION_PROBABILITY;
	unsigned int FACTOR_DOWNSCALE_X;
	unsigned int FACTOR_DOWNSCALE_Y;
	unsigned int FACTOR_DOWNSCALE_MULTIPLE;
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
