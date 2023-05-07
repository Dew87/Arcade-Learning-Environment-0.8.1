#include <conio.h>
#include <iostream>
#include "Program.hpp"

using namespace std;
using namespace NEAT;

Program::Program() : mIsRomLoaded(false), mALE(false), mAgent(NULL)
{
	load_neat_params("NEAT.ne");
	Config();
}

void Program::Config()
{
	// CONFIG load
	string temp;
	ifstream ifs("CONFIG", ios::in);
	ifs >> temp >> ALE_RANDOM_SEED;
	ifs >> temp >> ALE_REPEAT_ACTION_PROBABILITY;
	ifs >> temp >> FACTOR_DOWNSCALED_X;
	ifs >> temp >> FACTOR_DOWNSCALED_Y;
	ifs >> temp >> FACTOR_DOWNSCALED_MULTIPLE;
	ifs >> temp >> GENERATIONS;
	ifs >> temp >> MAXIMUM_NUMBER_OF_FRAMES;
	ifs >> temp >> PIXELS_X;
	ifs >> temp >> PIXELS_Y;
	ifs.close();

	// Calculations
	PIXELS_DOWNSCALED_X = PIXELS_X / FACTOR_DOWNSCALED_X;
	PIXELS_DOWNSCALED_Y = PIXELS_Y / FACTOR_DOWNSCALED_Y;
	DOWNSCALE_FACTOR = 1.0f / (float)(FACTOR_DOWNSCALED_X * FACTOR_DOWNSCALED_Y * FACTOR_DOWNSCALED_MULTIPLE);
	SENSOR_INPUTS = PIXELS_DOWNSCALED_X * PIXELS_DOWNSCALED_Y;

	// ALE
	mALE.setInt("random_seed", ALE_RANDOM_SEED);
	mALE.setFloat("repeat_action_probability", ALE_REPEAT_ACTION_PROBABILITY);
}

void Program::Info() const
{
	cout << "ALE NEAT made by David Erikssen\n";
	cout << "ALE_RANDOM_SEED = " << ALE_RANDOM_SEED << "\n";
	cout << "ALE_REPEAT_ACTION_PROBABILITY = " << ALE_REPEAT_ACTION_PROBABILITY << "\n";
	cout << "FACTOR_DOWNSCALED_X = " << FACTOR_DOWNSCALED_X << "\n";
	cout << "FACTOR_DOWNSCALED_Y = " << FACTOR_DOWNSCALED_Y << "\n";
	cout << "FACTOR_DOWNSCALED_MULTIPLE = " << FACTOR_DOWNSCALED_MULTIPLE << "\n";
	cout << "GENERATIONS = " << GENERATIONS << "\n";
	cout << "MAXIMUM_NUMBER_OF_FRAMES = " << MAXIMUM_NUMBER_OF_FRAMES << "\n";
	cout << "PIXELS_X = " << PIXELS_X << "\n";
	cout << "PIXELS_Y = " << PIXELS_Y << "\n";
	cout << "PIXELS_DOWNSCALED_X = " << PIXELS_DOWNSCALED_X << "\n";
	cout << "PIXELS_DOWNSCALED_Y = " << PIXELS_DOWNSCALED_Y << "\n";
	cout << "DOWNSCALE_FACTOR = " << DOWNSCALE_FACTOR << "\n";
	cout << "SENSOR_INPUTS = " << SENSOR_INPUTS << "\n\n";
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
	Genome *genome = new Genome(id, ifs);
	ifs.close();

	if (mAgent != NULL) { delete mAgent; }
	mAgent = new Organism(0.0, genome, 1);
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

	line = "CONFIG";
	log << line << "\n";
	ifs.open(line);
	while (getline(ifs, line)) { log << line << "\n"; }
	ifs.close();
	log << "\n";

	line = "NEAT.ne";
	log << line << "\n";
	ifs.open(line);
	while (getline(ifs, line)) { log << line << "\n"; }
	ifs.close();
	log << "\nLOG\n";
}

void Program::Play(size_t games, bool SDL)
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", SDL);
	mALE.setBool("sound", SDL);
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE.getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// NEAT
	while (mAgent == NULL) { LoadAgent(); }
	Network *network = mAgent->net;
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

	for (size_t i = 1; i <= games; i++)
	{
		mALE.reset_game();
		ale::reward_t totalReward = 0;
		while (!mALE.game_over() && mALE.getEpisodeFrameNumber() < MAXIMUM_NUMBER_OF_FRAMES)
		{
			// Get grayscale screen and downsample
			vector<unsigned char> grayscale_output_buffer;
			mALE.getScreenGrayscale(grayscale_output_buffer);
			vector<float> input = ProcessInput(grayscale_output_buffer);

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
		cout << "Game " << i << " score: " << totalReward << " time: " << mALE.getEpisodeFrameNumber() << "\n";
	}
	cout << "\n";
}

vector<float> Program::ProcessInput(const vector<unsigned char> &input) const
{
	vector<unsigned int> sum(SENSOR_INPUTS, 0);
	vector<float> output(SENSOR_INPUTS, 0.0f);

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
		if (j == FACTOR_DOWNSCALED_X)
		{
			j = 0;
			x++;
			if (x == PIXELS_DOWNSCALED_X)
			{
				x = 0;
				k++;
				if (k == FACTOR_DOWNSCALED_Y)
				{
					k = 0;
					y++;
				}
			}
		}
	}

	for (i = 0; i < sum.size(); i++)
	{
		output[i] = (float)sum[i] * DOWNSCALE_FACTOR;
	}

	return output;
}

void Program::Run()
{
	Info();
	while (true)
	{
		cout << "1: Play\n";
		cout << "2: Train\n";
		cout << "3: LoadAgent\n";
		cout << "4: LoadRom\n";
		cout << "5: LoadConfig\n";
		cout << "6: Info\n";
		cout << "7: Play100\n";
		cout << "Any other character to quit\n";

		char answer = _getch();
		cout << "\n";

		switch (answer)
		{
		case '1': Play(1, true); break;
		case '2': Train(); break;
		case '3': LoadAgent(); break;
		case '4': LoadRom(); break;
		case '5': Config(); break;
		case '6': Info(); break;
		case '7': Play(100, false); break;
		default: return;
		}
	}
}

void Program::Train()
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", false);
	mALE.setBool("sound", false);
	while (!mIsRomLoaded) { LoadRom(); }
	ale::ActionVect legal_actions = mALE.getLegalActionSet();
	cout << "Number of legal actions: " << legal_actions.size() << "\n";

	// NEAT
	cout << "\nNEAT\n";
	cout << "Spawning Population of Genome\n";
	Genome start_genome(SENSOR_INPUTS, (int)legal_actions.size(), 0, 0);
	Population population(&start_genome, NEAT::pop_size);
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

		Organism *champion = NULL;
		double championFitness = -DBL_MAX;
		double generationFitness = 0;

		for (vector<Organism*>::iterator i = population.organisms.begin(); i != population.organisms.end(); ++i)
		{
			Organism *organism = *i;
			Network *network = organism->net;

			mALE.reset_game();
			ale::reward_t totalReward = 0;
			while (!mALE.game_over() && mALE.getEpisodeFrameNumber() < MAXIMUM_NUMBER_OF_FRAMES)
			{
				// Get grayscale screen and downsample
				vector<unsigned char> grayscale_output_buffer;
				mALE.getScreenGrayscale(grayscale_output_buffer);
				vector<float> input = ProcessInput(grayscale_output_buffer);

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
			mAgent = new Organism(*champion);
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
