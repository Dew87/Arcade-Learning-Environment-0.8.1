#include "MonochromeScreen.hpp"
#include <SDL.h>

MonochromeScreen::MonochromeScreen(const char *title, int width, int height, int x, int y) : mX(x), mY(y)
{
	// Create SDL window
	window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_RESIZABLE);
	if (window == NULL)
	{
		SDL_Log("Unable to create window: %s", SDL_GetError());
	}

	// Create renderer
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == NULL)
	{
		SDL_Log("Unabel to create renderer: %s", SDL_GetError());
	}

	SDL_RenderSetLogicalSize(renderer, mX, mY);

	// Create texture
	texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, mX, mY);
	if (texture == NULL)
	{
		SDL_Log("Unable to create texture: %s", SDL_GetError());
	}
}

MonochromeScreen::~MonochromeScreen()
{
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
}

void MonochromeScreen::Render(const std::vector<unsigned char> &input)
{
	std::vector<unsigned char> output;
	output.reserve(input.size() * 4);

	for (std::vector<unsigned char>::const_iterator i = input.begin(); i != input.end(); i++)
	{
		output.push_back(*i);
		output.push_back(*i);
		output.push_back(*i);
		output.push_back(0);
	}

	// Update texture with new data
	int texture_pitch = 0;
	void *texture_pixels = NULL;
	if (SDL_LockTexture(texture, NULL, &texture_pixels, &texture_pitch) != 0)
	{
		SDL_Log("Unable to lock texture: %s", SDL_GetError());
	}
	else
	{
		SDL_memcpy(texture_pixels, &output[0], texture_pitch * mY);
	}
	SDL_UnlockTexture(texture);

	SDL_RenderClear(renderer);
	SDL_RenderCopy(renderer, texture, NULL, NULL);
	SDL_RenderPresent(renderer);
}
