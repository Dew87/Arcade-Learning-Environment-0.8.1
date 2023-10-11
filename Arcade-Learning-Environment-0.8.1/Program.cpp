#include "Program.hpp"
#include "ale_interface.hpp"
#include "MonochromeScreen.hpp"
#include "FNN/FNN.h"
#include "FNN/FNNPopulation.h"
#include "NEAT/population.h"
#include <conio.h>
#include <cmath>
#include <iostream>
#include <SDL.h>

using namespace std;

const char* CONFIG = "Config";
const char* CONFIG_FNN = "Config_FNN";
const char* CONFIG_NEAT = "Config_NEAT";

Program::Program() : mIsRomLoaded(false), mAgent(NULL), mALE(NULL), mInputScreen(NULL)
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
	Reset();
	SDL_Quit();
}

void Program::Config()
{
	// CONFIG load
	string temp;
	ifstream ifs(CONFIG);
	ifs >> temp >> SRAND_SEED;
	ifs >> temp >> PLAY_RUNS;
	ifs >> temp >> SCREEN;
	ifs >> temp >> SCREEN_OUTPUT;
	ifs >> temp >> ALE_FRAME_SKIP;
	ifs >> temp >> ALE_MAX_NUM_FRAMES_PER_EPISODE;
	ifs >> temp >> ALE_RANDOM_SEED;
	ifs >> temp >> ALE_REPEAT_ACTION_PROBABILITY;
	ifs >> temp >> FACTOR_DOWNSCALE_X;
	ifs >> temp >> FACTOR_DOWNSCALE_Y;
	ifs >> temp >> GENERATIONS;
	ifs >> temp >> PIXELS_X;
	ifs >> temp >> PIXELS_Y;
	ifs.close();

	// SRAND
	srand(SRAND_SEED);

	// Calculations
	CONVERSION_UCHAR_FLOAT = 1.0f / (float)(256);
	FACTOR_DOWNSCALE = FACTOR_DOWNSCALE_X * FACTOR_DOWNSCALE_Y;
	FACTOR_DOWNSCALE_INVERSE = 1.0f / (float)(FACTOR_DOWNSCALE);
	PIXELS_DOWNSCALED_X = PIXELS_X / FACTOR_DOWNSCALE_X;
	PIXELS_DOWNSCALED_Y = PIXELS_Y / FACTOR_DOWNSCALE_Y;
	PIXELS_DOWNSCALED = PIXELS_DOWNSCALED_X * PIXELS_DOWNSCALED_Y;

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
		float value = (float)(*i) * CONVERSION_UCHAR_FLOAT;
		output.push_back(value);
	}

	return output;
}

vector<unsigned char> Program::DownsampleOutputBuffer(const vector<unsigned char> &input) const
{
	vector<unsigned int> sum(PIXELS_DOWNSCALED, 0);
	vector<unsigned char> output;
	output.reserve(PIXELS_DOWNSCALED);

	for (size_t y = 0; y < PIXELS_DOWNSCALED_Y; ++y)
	{
		size_t sumOffsetY = y * PIXELS_DOWNSCALED_X;
		for (size_t x = 0; x < PIXELS_DOWNSCALED_X; ++x)
		{
			size_t sumOffset = x + sumOffsetY;
			size_t inputOffsetXY = (x * FACTOR_DOWNSCALE_X) + (y * FACTOR_DOWNSCALE_Y * PIXELS_X);
			for (size_t j = 0; j < FACTOR_DOWNSCALE_Y; ++j)
			{
				size_t inputOffset = (j * PIXELS_X) + inputOffsetXY;
				for (size_t i = 0; i < FACTOR_DOWNSCALE_X; ++i)
				{
					sum[sumOffset] += input[i + inputOffset];
				}
			}
		}
	}

	for (vector<unsigned int>::iterator i = sum.begin(); i != sum.end(); i++)
	{
		unsigned char value = (unsigned char)((*i) * FACTOR_DOWNSCALE_INVERSE);
		output.push_back(value);
	}

	return output;
}

void Program::InitializeALE(bool screen)
{
	mALE = new ale::ALEInterface(screen);

	mALE->setInt("frame_skip", ALE_FRAME_SKIP);
	mALE->setInt("max_num_frames_per_episode", ALE_MAX_NUM_FRAMES_PER_EPISODE);
	mALE->setInt("random_seed", ALE_RANDOM_SEED);
	mALE->setFloat("repeat_action_probability", ALE_REPEAT_ACTION_PROBABILITY);
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
	string str;
	getline(cin, str);
	if (mALE == NULL) { InitializeALE(false); }
	mALE->loadROM(str);
	mIsRomLoaded = true;
}

void Program::LogStart(ofstream &log)
{
	string str;
	ifstream ifs;

	str = CONFIG;
	log << "CONFIG" << "\n";
	ifs.open(str);
	while (getline(ifs, str)) { log << str << "\n"; }
	ifs.close();
	log << "\n";

	str = CONFIG_FNN;
	log << "FNN" << "\n";
	ifs.open(str);
	while (getline(ifs, str)) { log << str << "\n"; }
	ifs.close();
	log << "\n";

	str = CONFIG_NEAT;
	log << "NEAT" << "\n";
	ifs.open(str);
	while (getline(ifs, str)) { log << str << "\n"; }
	ifs.close();
	log << "\n";

	log << "LOG\n";
}

void Program::Play()
{
	// ALE
	cout << "\nALE\n";
	if (mALE == NULL) { InitializeALE(SCREEN); }
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE->getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// NEAT
	while (mAgent == NULL) { LoadAgent(); }
	NEAT::Network *network = mAgent->net;
	if (network->inputs.size() != PIXELS_DOWNSCALED)
	{
		cout << "Mismatch number of sensors " << network->inputs.size() << " expected " << PIXELS_DOWNSCALED << "\n\n";
		return;
	}
	if (network->outputs.size() != legal_actions.size())
	{
		cout << "Mismatch number of outputs " << network->outputs.size() << " expected " << legal_actions.size() << "\n\n";
		return;
	}

	// Create input screen
	if (SCREEN && SCREEN_OUTPUT && mInputScreen == NULL)
	{
		mInputScreen = new MonochromeScreen("Input", PIXELS_X, PIXELS_Y, PIXELS_DOWNSCALED_X, PIXELS_DOWNSCALED_Y);
	}

	for (size_t i = 1; i <= PLAY_RUNS; i++)
	{
		mALE->reset_game();
		ale::reward_t totalReward = 0;
		while (!mALE->game_over())
		{
			// Get grayscale screen and downsample
			vector<unsigned char> grayscale_output_buffer;
			mALE->getScreenGrayscale(grayscale_output_buffer);
			vector<unsigned char> downscaled_output = DownsampleOutputBuffer(grayscale_output_buffer);
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
			totalReward += mALE->act(action);
		}
		cout << "Game " << i << " score: " << totalReward << " time: " << mALE->getEpisodeFrameNumber() << "\n";
	}

	cout << "\n";
}

void Program::Print() const
{
	cout << "ALE Agent made by David Erikssen\n";
	cout << "SRAND_SEED = " << SRAND_SEED << "\n";
	cout << "PLAY_RUNS = " << PLAY_RUNS << "\n";
	cout << "SCREEN = " << SCREEN << "\n";
	cout << "SCREEN_OUTPUT = " << SCREEN_OUTPUT << "\n";
	cout << "ALE_FRAME_SKIP = " << ALE_FRAME_SKIP << "\n";
	cout << "ALE_MAX_NUM_FRAMES_PER_EPISODE = " << ALE_MAX_NUM_FRAMES_PER_EPISODE << "\n";
	cout << "ALE_RANDOM_SEED = " << ALE_RANDOM_SEED << "\n";
	cout << "ALE_REPEAT_ACTION_PROBABILITY = " << ALE_REPEAT_ACTION_PROBABILITY << "\n";
	cout << "FACTOR_DOWNSCALE_X = " << FACTOR_DOWNSCALE_X << "\n";
	cout << "FACTOR_DOWNSCALE_Y = " << FACTOR_DOWNSCALE_Y << "\n";
	cout << "GENERATIONS = " << GENERATIONS << "\n";
	cout << "PIXELS_X = " << PIXELS_X << "\n";
	cout << "PIXELS_Y = " << PIXELS_Y << "\n";
	cout << "CONVERSION_UCHAR_FLOAT = " << CONVERSION_UCHAR_FLOAT << "\n";
	cout << "FACTOR_DOWNSCALE_INVERSE = " << FACTOR_DOWNSCALE_INVERSE << "\n";
	cout << "PIXELS_DOWNSCALED = " << PIXELS_DOWNSCALED << "\n";
	cout << "PIXELS_DOWNSCALED_X = " << PIXELS_DOWNSCALED_X << "\n";
	cout << "PIXELS_DOWNSCALED_Y = " << PIXELS_DOWNSCALED_Y << "\n";

	cout << "\nFNN\n";
	FNN::print_fnn_params();

	cout << "\nNEAT\n";
	NEAT::print_neat_params();

	cout << "\n";
}

void Program::Reset()
{
	mIsRomLoaded = false;
	if (mAgent != NULL)
	{
		delete mAgent;
		mAgent = NULL;
	}
	if (mALE != NULL)
	{
		delete mALE;
		mALE = NULL;
	}
	if (mInputScreen != NULL)
	{
		delete mInputScreen;
		mInputScreen = NULL;
	}
}

void Program::Run()
{
	Print();
	while (true)
	{
		cout << "1: Play\n";
		cout << "2: Train FNN\n";
		cout << "3: Train NEAT\n";
		cout << "4: LoadAgent\n";
		cout << "5: LoadRom\n";
		cout << "6: LoadConfig\n";
		cout << "7: PrintConfig\n";
		cout << "8: Reset\n";
		cout << "Any other character to quit\n";

		char answer = _getch();
		cout << "\n";

		switch (answer)
		{
		case '1': Play(); break;
		case '2': TrainFNN(); break;
		case '3': TrainNEAT(); break;
		case '4': LoadAgent(); cout << "\n"; break;
		case '5': LoadRom();  cout << "\n"; break;
		case '6': Config(); break;
		case '7': Print(); break;
		case '8': Reset(); break;
		default: return;
		}
	}
}

void Program::TrainFNN()
{
	// Don't run if pop is 0 or below
	if (FNN::pop_size <= 0) { return; }

	// ALE
	cout << "\nALE\n";
	if (mALE == NULL) { InitializeALE(false); }
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE->getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// FNN
	cout << "\nFNN\n";
	cout << "Spawning Population of Genome\n";
	NEAT::Genome start_genome(PIXELS_DOWNSCALED, (int)legal_actions.size(), FNN::hidden_nodes, 3);
	FNN::FNNPopulation population(&start_genome, FNN::pop_size);
	cout << "Verifying Spawned Pop\n";
	population.verify();

	// Input agent file name
	cout << "Save agent to file: ";
	string filename;
	getline(cin, filename);

	// Log file open
	ofstream log(filename + ".txt", ofstream::app);
	LogStart(log);

	// Reset agent
	if (mAgent != NULL) { delete mAgent; mAgent = NULL; }
	double agentFitness = -DBL_MAX;

	time_t start = time(NULL);
	for (size_t generation = 1; generation <= GENERATIONS; generation++)
	{
		cout << "Generation " << generation << "\n";
		double fitnessSum = 0.0;

		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			NEAT::Organism *organism = *i;
			NEAT::Network *network = organism->net;

			mALE->reset_game();
			ale::reward_t totalReward = 0;
			while (!mALE->game_over())
			{
				// Get grayscale screen and downsample into input
				vector<unsigned char> grayscale_output_buffer;
				mALE->getScreenGrayscale(grayscale_output_buffer);
				vector<unsigned char> downscaled_output = DownsampleOutputBuffer(grayscale_output_buffer);
				vector<float> input = ConvertOutputBuffer(downscaled_output);

				// Input sensor data
				network->load_sensors(input);
				network->activate();

				// Find and perform highest output action
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
				totalReward += mALE->act(action);
			}

			// Fitness organism
			double fitness = (double)totalReward;
			organism->fitness = fitness;
			fitnessSum += fitness;

			// Organism output
			//cout << "Organism " << (organism->gnome)->genome_id << " fitness: " << organism->fitness << " time: " << mALE->getEpisodeFrameNumber() << "\n";
		}

		// Sort population on fitness and get champion
		sort(population.organisms.begin(), population.organisms.end(), NEAT::order_orgs);
		NEAT::Organism *champion = *population.organisms.begin();

		// Replace agent with champion if champion is better
		if (agentFitness < champion->fitness)
		{
			if (mAgent != NULL) { delete mAgent; }
			mAgent = new NEAT::Organism(*champion);
			agentFitness = champion->fitness;
		}

		// Fitness average and deviation
		double fitnessAverage = fitnessSum / (double)population.organisms.size();
		double sumSquared = 0.0;
		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			sumSquared += ((*i)->fitness - fitnessAverage) * ((*i)->fitness - fitnessAverage);
		}
		double fitnessDeviation = sqrt(sumSquared / (double)population.organisms.size());

		// Log output
		log << "Generation " << generation << ": fitness average " << fitnessAverage << " deviation " << fitnessDeviation << " max " << champion->fitness << "\n";

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
	// Don't run if pop is 0 or below
	if (NEAT::pop_size <= 0) { return; }

	// ALE
	cout << "\nALE\n";
	if (mALE == NULL) { InitializeALE(false); }
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE->getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// NEAT
	cout << "\nNEAT\n";
	cout << "Spawning Population of Genome\n";
	NEAT::Genome start_genome(PIXELS_DOWNSCALED, (int)legal_actions.size(), 0, 0);
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

	// Reset agent
	if (mAgent != NULL) { delete mAgent; mAgent = NULL; }
	double agentFitness = -DBL_MAX;

	time_t start = time(NULL);
	for (size_t generation = 1; generation <= GENERATIONS; generation++)
	{
		cout << "Generation " << generation << "\n";
		double fitnessSum = 0;

		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			NEAT::Organism *organism = *i;
			NEAT::Network *network = organism->net;

			mALE->reset_game();
			ale::reward_t totalReward = 0;
			while (!mALE->game_over())
			{
				// Get grayscale screen and downsample into input
				vector<unsigned char> grayscale_output_buffer;
				mALE->getScreenGrayscale(grayscale_output_buffer);
				vector<unsigned char> downscaled_output = DownsampleOutputBuffer(grayscale_output_buffer);
				vector<float> input = ConvertOutputBuffer(downscaled_output);

				// Input sensor data
				network->load_sensors(input);
				network->activate();

				// Find and perform highest output action
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
				totalReward += mALE->act(action);
			}

			// Fitness organism
			double fitness = (double)totalReward;
			organism->fitness = fitness;
			fitnessSum += fitness;

			// Organism output
			//cout << "Organism " << (organism->gnome)->genome_id << " fitness: " << organism->fitness << " time: " << mALE->getEpisodeFrameNumber() << "\n";
		}

		// Sort population on fitness and get champion
		sort(population.organisms.begin(), population.organisms.end(), NEAT::order_orgs);
		NEAT::Organism *champion = *population.organisms.begin();

		// Replace agent with champion if champion is better
		if (agentFitness < champion->fitness)
		{
			if (mAgent != NULL) { delete mAgent; }
			mAgent = new NEAT::Organism(*champion);
			agentFitness = champion->fitness;
		}

		// Fitness average and deviation
		double fitnessAverage = fitnessSum / (double)population.organisms.size();
		double sumSquared = 0.0;
		for (vector<NEAT::Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			sumSquared += ((*i)->fitness - fitnessAverage) * ((*i)->fitness - fitnessAverage);
		}
		double fitnessDeviation = sqrt(sumSquared / (double)population.organisms.size());

		// Log output
		log << "Generation " << generation << " species[" << population.species.size() << "]: fitness average " << fitnessAverage << " deviation " << fitnessDeviation << " max " << champion->fitness << "\n";

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
