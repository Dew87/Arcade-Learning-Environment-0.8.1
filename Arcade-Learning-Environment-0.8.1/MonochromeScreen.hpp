#ifndef MONOCHROME_SCREEN
#define MONOCHROME_SCREEN

#include <vector>

class SDL_Renderer;
class SDL_Texture;
class SDL_Window;

class MonochromeScreen
{
public:
	MonochromeScreen(const char *title, int width, int height);
	~MonochromeScreen();
	void Render(const std::vector<unsigned char> &input);

private:
	SDL_Renderer *renderer;
	SDL_Texture *texture;
	SDL_Window *window;

	int mWidth;
	int mHeight;
};

#endif
