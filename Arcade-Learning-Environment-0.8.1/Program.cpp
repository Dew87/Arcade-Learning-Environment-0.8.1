#include <conio.h>
#include <iostream>
#include <SDL.h>
#include "Program.hpp"
#include "MonochromeScreen.hpp"
#include "FNN/FNN.h"
#include "FNN/FNNPopulation.h"
#include "NEAT/population.h"

using namespace std;

const char* CONFIG = "Config";
const char* CONFIG_FNN = "Config_FNN";
const char* CONFIG_NEAT = "Config_NEAT";

Program::Program() : mIsRomLoaded(false), mALE(false), mAgent(NULL), mInputScreen(NULL)
{
	// SDL init
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
	}

	Config();
}

Program::~Program()
{
	if (mInputScreen != NULL)
	{
		delete mInputScreen;
		mInputScreen = NULL;
	}

	SDL_Quit();
}

void Program::Config()
{
	// CONFIG load
	string temp;
	ifstream ifs(CONFIG);
	ifs >> temp >> SRAND_SEED;
	ifs >> temp >> ALE_MAX_NUM_FRAMES_PER_EPISODE;
	ifs >> temp >> ALE_RANDOM_SEED;
	ifs >> temp >> ALE_REPEAT_ACTION_PROBABILITY;
	ifs >> temp >> FACTOR_DOWNSCALE_X;
	ifs >> temp >> FACTOR_DOWNSCALE_Y;
	ifs >> temp >> FACTOR_DOWNSCALE_MULTIPLE;
	ifs >> temp >> GENERATIONS;
	ifs >> temp >> PIXELS_X;
	ifs >> temp >> PIXELS_Y;
	ifs.close();

	// SRAND
	srand(SRAND_SEED);

	// Calculations
	PIXELS_DOWNSCALED_X = PIXELS_X / FACTOR_DOWNSCALE_X;
	PIXELS_DOWNSCALED_Y = PIXELS_Y / FACTOR_DOWNSCALE_Y;
	DOWNSCALE_FACTOR = 1.0f / (float)(FACTOR_DOWNSCALE_X * FACTOR_DOWNSCALE_Y);
	DOWNSCALE_MULTIPLE = 1.0f / (float)(FACTOR_DOWNSCALE_MULTIPLE);
	SENSOR_INPUTS = PIXELS_DOWNSCALED_X * PIXELS_DOWNSCALED_Y;

	// ALE
	mALE.setInt("max_num_frames_per_episode", ALE_MAX_NUM_FRAMES_PER_EPISODE);
	mALE.setInt("random_seed", ALE_RANDOM_SEED);
	mALE.setFloat("repeat_action_probability", ALE_REPEAT_ACTION_PROBABILITY);

	// FNN
	FNN::load_fnn_params(CONFIG_FNN);

	// NEAT
	NEAT::load_neat_params(CONFIG_NEAT);
}

vector<float> Program::ConvertOutputBuffer(const vector<unsigned char> &input) const
{
	vector<float> output;
	output.reserve(input.size());

	for (vector<unsigned char>::const_iterator i = input.begin(); i != input.end(); i++)
	{
		float value = (float)(*i) * DOWNSCALE_MULTIPLE;
		output.push_back(value);
	}

	return output;
}

vector<unsigned char> Program::DownscaleOutputBuffer(const vector<unsigned char> &input) const
{
	vector<unsigned int> sum(SENSOR_INPUTS, 0);
	vector<unsigned char> output;
	output.reserve(SENSOR_INPUTS);

	size_t i = 0;
	size_t j = 0;
	size_t k = 0;
	size_t x = 0;
	size_t y = 0;
	while (i < input.size())
	{
		sum[x + (y * PIXELS_DOWNSCALED_X)] += input[i];
		i++;
		j++;
		if (j == FACTOR_DOWNSCALE_X)
		{
			j = 0;
			x++;
			if (x == PIXELS_DOWNSCALED_X)
			{
				x = 0;
				k++;
				if (k == FACTOR_DOWNSCALE_Y)
				{
					k = 0;
					y++;
				}
			}
		}
	}

	for (vector<unsigned int>::iterator i = sum.begin(); i != sum.end(); i++)
	{
		unsigned char value = (unsigned char)((*i) * DOWNSCALE_FACTOR);
		output.push_back(value);
	}

	return output;
}

void Program::LoadAgent()
{
	cout << "Input agent file name: ";
	string str;
	getline(cin, str);

	// Read from file
	int id;
	ifstream ifs(str, ios::in);
	ifs >> str >> id;
	cout << "Reading in Genome id " << id << endl;
	NEAT::Genome *genome = new NEAT::Genome(id, ifs);
	ifs.close();

	if (mAgent != NULL) { delete mAgent; }
	mAgent = new NEAT::Organism(0.0, genome, 1);
}

void Program::LoadRom()
{
	cout << "Input rom file name: ";
	string rom;
	getline(cin, rom);
	mALE.loadROM(rom);
	mIsRomLoaded = true;
}

void Program::LogStart(ofstream &log)
{
	string line;
	ifstream ifs;

	line = CONFIG;
	log << "CONFIG" << "\n";
	ifs.open(line);
	while (getline(ifs, line)) { log << line << "\n"; }
	ifs.close();
	log << "\n";

	line = CONFIG_FNN;
	log << "FNN" << "\n";
	ifs.open(line);
	while (getline(ifs, line)) { log << line << "\n"; }
	ifs.close();
	log << "\n";

	line = CONFIG_NEAT;
	log << "NEAT" << "\n";
	ifs.open(line);
	while (getline(ifs, line)) { log << line << "\n"; }
	ifs.close();
	log << "\n";

	log << "LOG\n";
}

void Program::Play(size_t games, bool screen)
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", screen);
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE.getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// NEAT
	while (mAgent == NULL) { LoadAgent(); }
	NEAT::Network *network = mAgent->net;
	if (network->inputs.size() != SENSOR_INPUTS)
	{
		cout << "Mismatch number of sensors: " << network->inputs.size() << "//" << SENSOR_INPUTS << "\n";
		return;
	}
	if (network->outputs.size() != legal_actions.size())
	{
		cout << "Mismatch number of outputs: " << network->outputs.size() << "//" << legal_actions.size() << "\n";
		return;
	}

	// Create input screen
	if (screen && mInputScreen == NULL)
	{
		mInputScreen = new MonochromeScreen("Input", PIXELS_DOWNSCALED_X, PIXELS_DOWNSCALED_Y);
	}

	for (size_t i = 1; i <= games; i++)
	{
		mALE.reset_game();
		ale::reward_t totalReward = 0;
		while (!mALE.game_over())
		{
			// Get grayscale screen and downsample
			vector<unsigned char> grayscale_output_buffer;
			mALE.getScreenGrayscale(grayscale_output_buffer);
			vector<unsigned char> downscaled_output = DownscaleOutputBuffer(grayscale_output_buffer);
			vector<float> input = ConvertOutputBuffer(downscaled_output);

			// Input sensor data and read output
			network->load_sensors(input);
			network->activate();

			// Display input
			if (mInputScreen)
			{
				mInputScreen->Render(downscaled_output);
			}

			size_t highestActivationIndex = 0;
			double highestActivationValue = -DBL_MAX;
			for (size_t j = 0; j < network->outputs.size(); j++)
			{
				double activation = network->outputs[j]->activation;
				if (highestActivationValue < activation)
				{
					highestActivationIndex = j;
					highestActivationValue = activation;
				}
			}

			ale::Action action = legal_actions[highestActivationIndex];
			totalReward += mALE.act(action);
		}
		cout << "Game " << i << " score: " << totalReward << " time: " << mALE.getEpisodeFrameNumber() << "\n";
	}

	cout << "\n";
}

void Program::Print() const
{
	cout << "ALE Agent made by David Erikssen\n";
	cout << "SRAND_SEED = " << SRAND_SEED << "\n";
	cout << "ALE_MAX_NUM_FRAMES_PER_EPISODE = " << ALE_MAX_NUM_FRAMES_PER_EPISODE << "\n";
	cout << "ALE_RANDOM_SEED = " << ALE_RANDOM_SEED << "\n";
	cout << "ALE_REPEAT_ACTION_PROBABILITY = " << ALE_REPEAT_ACTION_PROBABILITY << "\n";
	cout << "FACTOR_DOWNSCALE_X = " << FACTOR_DOWNSCALE_X << "\n";
	cout << "FACTOR_DOWNSCALE_Y = " << FACTOR_DOWNSCALE_Y << "\n";
	cout << "FACTOR_DOWNSCALE_MULTIPLE = " << FACTOR_DOWNSCALE_MULTIPLE << "\n";
	cout << "GENERATIONS = " << GENERATIONS << "\n";
	cout << "PIXELS_X = " << PIXELS_X << "\n";
	cout << "PIXELS_Y = " << PIXELS_Y << "\n";
	cout << "PIXELS_DOWNSCALED_X = " << PIXELS_DOWNSCALED_X << "\n";
	cout << "PIXELS_DOWNSCALED_Y = " << PIXELS_DOWNSCALED_Y << "\n";
	cout << "DOWNSCALE_FACTOR = " << DOWNSCALE_FACTOR << "\n";
	cout << "DOWNSCALE_MULTIPLE = " << DOWNSCALE_MULTIPLE << "\n";
	cout << "SENSOR_INPUTS = " << SENSOR_INPUTS << "\n";

	cout << "\nFNN\n";
	FNN::print_fnn_params();

	cout << "\nNEAT\n";
	NEAT::print_neat_params();

	cout << "\n";
}

void Program::Run()
{
	Print();
	while (true)
	{
		cout << "1: Play\n";
		cout << "2: Train NEAT\n";
		cout << "3: Train FNN\n";
		cout << "4: LoadAgent\n";
		cout << "5: LoadRom\n";
		cout << "6: LoadConfig\n";
		cout << "7: PrintConfig\n";
		cout << "8: Play100\n";
		cout << "Any other character to quit\n";

		char answer = _getch();
		cout << "\n";

		switch (answer)
		{
		case '1': Play(1, true); break;
		case '2': TrainNEAT(); break;
		case '3': TrainFNN(); break;
		case '4': LoadAgent(); break;
		case '5': LoadRom(); break;
		case '6': Config(); break;
		case '7': Print(); break;
		case '8': Play(100, false); break;
		default: return;
		}
	}
}

void Program::TrainFNN()
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", false);
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE.getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	cout << "\nFNN\n";
	cout << "Spawning Population of Genome\n";
	FNN::FNNPopulation population(SENSOR_INPUTS, (int)legal_actions.size(), FNN::hidden_nodes, FNN::pop_size);
	cout << "Verifying Spawned Pop\n";
	population.verify();

	// Input agent file name
	cout << "Save agent to file: ";
	string filename;
	getline(cin, filename);

	// Log file open
	ofstream log(filename + ".txt", ofstream::app);
	LogStart(log);

	if (mAgent != NULL) { delete mAgent; }
	mAgent = NULL;
	double agentFitness = -DBL_MAX;

	time_t start = time(NULL);
	for (size_t generation = 1; generation <= GENERATIONS; generation++)
	{
		cout << "Generation " << generation << "\n";

		NEAT::Organism *champion = NULL;
		double championFitness = -DBL_MAX;
		double generationFitness = 0;

		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			NEAT::Organism *organism = *i;
			NEAT::Network *network = organism->net;

			mALE.reset_game();
			ale::reward_t totalReward = 0;
			while (!mALE.game_over())
			{
				// Get grayscale screen and downsample
				vector<unsigned char> grayscale_output_buffer;
				mALE.getScreenGrayscale(grayscale_output_buffer);
				vector<unsigned char> downscaled_output = DownscaleOutputBuffer(grayscale_output_buffer);
				vector<float> input = ConvertOutputBuffer(downscaled_output);

				// Input sensor data and read output
				network->load_sensors(input);
				network->activate();

				size_t highestActivationIndex = 0;
				double highestActivationValue = -DBL_MAX;
				for (size_t j = 0; j < network->outputs.size(); j++)
				{
					double activation = network->outputs[j]->activation;
					if (highestActivationValue < activation)
					{
						highestActivationIndex = j;
						highestActivationValue = activation;
					}
				}

				ale::Action action = legal_actions[highestActivationIndex];
				totalReward += mALE.act(action);
			}
			organism->fitness = (double)totalReward;
			generationFitness += (double)totalReward;
			//cout << "Organism " << (organism->gnome)->genome_id << " fitness: " << organism->fitness << " time: " << mALE.getEpisodeFrameNumber() << "\n";

			if (championFitness < organism->fitness)
			{
				champion = organism;
				championFitness = organism->fitness;
			}
		}

		// New champion. If true copy champion to agent
		if (agentFitness < championFitness)
		{
			if (mAgent != NULL) { delete mAgent; }
			mAgent = new NEAT::Organism(*champion);
			agentFitness = championFitness;
		}

		// Log output
		log << "Generation " << generation << ": fitness average " << generationFitness / (double)FNN::pop_size << " max " << championFitness << "\n";

		// Create the next generation
		population.epoch((int)generation);
	}
	cout << "Elapsed time: " << time(NULL) - start << "\n";

	// Write agent to file
	if (mAgent != NULL)
	{
		ofstream ofs(filename, ofstream::app);
		mAgent->gnome->print_to_file(ofs);
		ofs.close();
		cout << "Agent saved to file: " << filename << "\n";
	}

	// Log file close
	log.close();

	cout << "\n";
}

void Program::TrainNEAT()
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", false);
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE.getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	cout << "\nNEAT\n";
	cout << "Spawning Population of Genome\n";
	NEAT::Genome start_genome(SENSOR_INPUTS, (int)legal_actions.size(), 0, 0);
	NEAT::Population population(&start_genome, NEAT::pop_size);
	cout << "Verifying Spawned Pop\n";
	population.verify();

	// Input agent file name
	cout << "Save agent to file: ";
	string filename;
	getline(cin, filename);

	// Log file open
	ofstream log(filename + ".txt", ofstream::app);
	LogStart(log);

	if (mAgent != NULL) { delete mAgent; }
	mAgent = NULL;
	double agentFitness = -DBL_MAX;

	time_t start = time(NULL);
	for (size_t generation = 1; generation <= GENERATIONS; generation++)
	{
		cout << "Generation " << generation << "\n";

		NEAT::Organism *champion = NULL;
		double championFitness = -DBL_MAX;
		double generationFitness = 0;

		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			NEAT::Organism *organism = *i;
			NEAT::Network *network = organism->net;

			mALE.reset_game();
			ale::reward_t totalReward = 0;
			while (!mALE.game_over())
			{
				// Get grayscale screen and downsample
				vector<unsigned char> grayscale_output_buffer;
				mALE.getScreenGrayscale(grayscale_output_buffer);
				vector<unsigned char> downscaled_output = DownscaleOutputBuffer(grayscale_output_buffer);
				vector<float> input = ConvertOutputBuffer(downscaled_output);

				// Input sensor data and read output
				network->load_sensors(input);
				network->activate();

				size_t highestActivationIndex = 0;
				double highestActivationValue = -DBL_MAX;
				for (size_t j = 0; j < network->outputs.size(); j++)
				{
					double activation = network->outputs[j]->activation;
					if (highestActivationValue < activation)
					{
						highestActivationIndex = j;
						highestActivationValue = activation;
					}
				}

				ale::Action action = legal_actions[highestActivationIndex];
				totalReward += mALE.act(action);
			}
			organism->fitness = (double)totalReward;
			generationFitness += (double)totalReward;
			//cout << "Organism " << (organism->gnome)->genome_id << " fitness: " << organism->fitness << " time: " << mALE.getEpisodeFrameNumber() << "\n";

			if (championFitness < organism->fitness)
			{
				champion = organism;
				championFitness = organism->fitness;
			}
		}

		// New champion. If true copy champion to agent
		if (agentFitness < championFitness)
		{
			if (mAgent != NULL) { delete mAgent; }
			mAgent = new NEAT::Organism(*champion);
			agentFitness = championFitness;
		}

		// Log output
		log << "Generation " << generation << " species[" << population.species.size() << "]: fitness average " << generationFitness / (double)NEAT::pop_size << " max " << championFitness << "\n";

		// Create the next generation
		population.epoch((int)generation);
	}
	cout << "Elapsed time: " << time(NULL) - start << "\n";

	// Write agent to file
	if (mAgent != NULL)
	{
		ofstream ofs(filename, ofstream::app);
		mAgent->gnome->print_to_file(ofs);
		ofs.close();
		cout << "Agent saved to file: " << filename << "\n";
	}

	// Log file close
	log.close();

	cout << "\n";
}
