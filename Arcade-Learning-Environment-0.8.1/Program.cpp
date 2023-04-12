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
	ifstream iFile("CONFIG", ios::in);
	iFile >> temp >> ALE_RANDOM_SEED;
	iFile >> temp >> FACTOR_DOWNSCALED_X;
	iFile >> temp >> FACTOR_DOWNSCALED_Y;
	iFile >> temp >> FACTOR_DOWNSCALED_MULTIPLE;
	iFile >> temp >> GENERATIONS;
	iFile >> temp >> MAXIMUM_NUMBER_OF_FRAMES;
	iFile >> temp >> PIXELS_X;
	iFile >> temp >> PIXELS_Y;
	iFile.close();

	// Calculations
	PIXELS_DOWNSCALED_X = PIXELS_X / FACTOR_DOWNSCALED_X;
	PIXELS_DOWNSCALED_Y = PIXELS_Y / FACTOR_DOWNSCALED_Y;
	DOWNSCALE_FACTOR = 1.0f / (float)(FACTOR_DOWNSCALED_X * FACTOR_DOWNSCALED_Y * FACTOR_DOWNSCALED_MULTIPLE);
	SENSOR_INPUTS = PIXELS_DOWNSCALED_X * PIXELS_DOWNSCALED_Y;

	// ALE
	mALE.setInt("random_seed", ALE_RANDOM_SEED);
}

void Program::Info() const
{
	cout << "ALE NEAT made by David Erikssen\n";
	cout << "ALE_RANDOM_SEED = " << ALE_RANDOM_SEED << "\n";
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
	ifstream iFile(str, ios::in);
	iFile >> str >> id;
	cout << "Reading in Genome id " << id << endl;
	Genome *genome = new Genome(id, iFile);
	iFile.close();

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

void Program::Play()
{
	// ALE
	cout << "\nALE\n";
	mALE.setBool("display_screen", true);
	mALE.setBool("sound", true);
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
	cout << "Agent score: " << totalReward << " time: " << mALE.getEpisodeFrameNumber() << "\n\n";
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
		cout << "1: Play" << endl;
		cout << "2: Train" << endl;
		cout << "3: LoadAgent" << endl;
		cout << "4: LoadRom" << endl;
		cout << "5: LoadConfig" << endl;
		cout << "6: Info" << endl;
		cout << "Any other character to quit" << endl;

		char answer = _getch();
		cout << endl;

		switch (answer)
		{
		case '1': Play(); break;
		case '2': Train(); break;
		case '3': LoadAgent(); break;
		case '4': LoadRom(); break;
		case '5': Config(); break;
		case '6': Info(); break;
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

	if (mAgent != NULL) { delete mAgent; }
	mAgent = NULL;
	double agentFitness = -DBL_MAX;

	time_t start = time(NULL);
	for (size_t generation = 1; generation <= GENERATIONS; generation++)
	{
		cout << "Generation " << generation << "\n";

		Organism *champion = NULL;
		double championFitness = -DBL_MAX;

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
			cout << "Organism " << (organism->gnome)->genome_id << " fitness: " << organism->fitness << " time: " << mALE.getEpisodeFrameNumber() << "\n";

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

		//Create the next generation
		population.epoch((int)generation);
	}
	cout << "Elapsed time: " << time(NULL) - start << "\n\n";

	// Write agent to file
	if (mAgent != NULL)
	{
		ofstream ofs(filename, ofstream::out);
		mAgent->gnome->print_to_file(ofs);
	}
}
