#ifndef PROGRAM_HPP
#define PROGRAM_HPP

#include<ostream>
#include<vector>

namespace ale { class ALEInterface; }
namespace NEAT { class Organism; }
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
	void InitializeALE(bool screen);
	void LoadAgent();
	void LoadRom();
	void LogStart(std::ofstream &log);
	void Play();
	void Print() const;
	void Reset();
	void TrainFNN();
	void TrainNEAT();

	// CONFIG
	bool SCREEN;
	bool SCREEN_OUTPUT;

	float ALE_REPEAT_ACTION_PROBABILITY;
	float CONVERSION_UCHAR_FLOAT;
	float FACTOR_DOWNSCALE_INVERSE;

	unsigned int ALE_FRAME_SKIP;
	unsigned int ALE_MAX_NUM_FRAMES_PER_EPISODE;
	unsigned int ALE_RANDOM_SEED;
	unsigned int FACTOR_DOWNSCALE;
	unsigned int FACTOR_DOWNSCALE_X;
	unsigned int FACTOR_DOWNSCALE_Y;
	unsigned int GENERATIONS;
	unsigned int PIXELS_DOWNSCALED;
	unsigned int PIXELS_DOWNSCALED_X;
	unsigned int PIXELS_DOWNSCALED_Y;
	unsigned int PIXELS_X;
	unsigned int PIXELS_Y;
	unsigned int PLAY_RUNS;
	unsigned int SRAND_SEED;

	// Variables
	bool mIsRomLoaded;
	NEAT::Organism *mAgent;
	ale::ALEInterface *mALE;
	MonochromeScreen *mInputScreen;
};

#endif
