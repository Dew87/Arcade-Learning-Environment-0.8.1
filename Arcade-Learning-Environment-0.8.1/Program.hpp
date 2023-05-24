#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include "ale_interface.hpp"
#include "NEAT/organism.h"

class MonochromeScreen;

class Program
{
public:
	Program();
	~Program();
	void Run();

private:
	// Functions
	void Config();
	std::vector<float> ConvertOutputBuffer(const std::vector<unsigned char> &input) const;
	std::vector<unsigned char> DownsampleOutputBuffer(const std::vector<unsigned char> &input) const;
	void LoadAgent();
	void LoadRom();
	void LogStart(std::ofstream &log);
	void Play(size_t runs, bool screen);
	void Print() const;
	void TrainFNN();
	void TrainNEAT();

	// CONFIG
	unsigned int SRAND_SEED;
	unsigned int ALE_RANDOM_SEED;
	float ALE_REPEAT_ACTION_PROBABILITY;
	unsigned int FACTOR_DOWNSCALE_X;
	unsigned int FACTOR_DOWNSCALE_Y;
	unsigned int FACTOR_DOWNSCALE_MULTIPLE;
	unsigned int GENERATIONS;
	unsigned int ALE_MAX_NUM_FRAMES_PER_EPISODE;
	unsigned int PIXELS_X;
	unsigned int PIXELS_Y;
	unsigned int PIXELS_DOWNSCALED_X;
	unsigned int PIXELS_DOWNSCALED_Y;
	float DOWNSCALE_FACTOR;
	float DOWNSCALE_MULTIPLE;
	unsigned int SENSOR_INPUTS;

	// Variables
	bool mIsRomLoaded;
	ale::ALEInterface mALE;
	NEAT::Organism *mAgent;
	MonochromeScreen *mInputScreen;
};

#endif
