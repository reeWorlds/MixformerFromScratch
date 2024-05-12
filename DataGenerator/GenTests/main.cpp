#include <noise.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <array>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace noise;


class Config
{
public:

	int n;
	int m;

	int seed;
	float scale;
	int octaves;
	float persistance;
	float lacunarity;

	Config(int _n, int _m, int _seed, float _scale, int _octaves,
		float _persistance, float _lacunarity)
	{
		n = _n;
		m = _m;

		seed = _seed;
		scale = _scale;
		octaves = _octaves;
		persistance = _persistance;
		lacunarity = _lacunarity;
	}
};


vector <vector <float> > genNoise(Config config)
{
	vector <vector <float> > img(config.n, vector <float>(config.m));

	module::Perlin noise;

	noise.SetSeed(config.seed);
	noise.SetFrequency(1.0 / config.scale);
	noise.SetOctaveCount(config.octaves);
	noise.SetPersistence(config.persistance);
	noise.SetLacunarity(config.lacunarity);

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			img[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			img[i][j] = max(0.0f, min(1.0f, img[i][j]));
		}
	}

	return img;
}


cv::Mat genImage1(Config config)
{
	vector<vector<float> > noise_img = genNoise(config);
	vector<vector<int> > noiseType(config.n, vector<int>(config.m));

	float water_threshold = 0.24;
	float sand_threshold = 0.4;
	float grass_threshold = 0.6;
	float rock_threshold = 0.76;
	int base_colors[5][3] =
	{
		{ 205, 0, 0 },
		{ 138, 188, 204 },
		{ 60, 245, 25 },
		{ 80, 80, 80 },
		{ 225, 225, 255 }
	};
	int shift_colors[5][3] =
	{
		{ 40, 0, 0 },
		{ -20, -20, -20 },
		{ -20, -50, -10 },
		{ 40, 40, 40 },
		{ 30, 30, 0 }
	};

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			if (noise_img[i][j] < water_threshold)
			{
				noiseType[i][j] = 0;
				noise_img[i][j] = (noise_img[i][j] - 0.0) / (water_threshold - 0.0);
			}
			else if (noise_img[i][j] < sand_threshold)
			{
				noiseType[i][j] = 1;
				noise_img[i][j] = (noise_img[i][j] - water_threshold) / (sand_threshold - water_threshold);
			}
			else if (noise_img[i][j] < grass_threshold)
			{
				noiseType[i][j] = 2;
				noise_img[i][j] = (noise_img[i][j] - sand_threshold) / (grass_threshold - sand_threshold);
			}
			else if (noise_img[i][j] < rock_threshold)
			{
				noiseType[i][j] = 3;
				noise_img[i][j] = (noise_img[i][j] - grass_threshold) / (rock_threshold - grass_threshold);
			}
			else
			{
				noiseType[i][j] = 4;
				noise_img[i][j] = (noise_img[i][j] - rock_threshold) / (1.0 - rock_threshold);
			}
		}
	}

	cv::Mat img(config.n, config.m, CV_8UC3);

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			int my_type = noiseType[i][j];
			uchar cols[3] =
			{
				base_colors[my_type][0] + shift_colors[my_type][0] * noise_img[i][j],
				base_colors[my_type][1] + shift_colors[my_type][1] * noise_img[i][j],
				base_colors[my_type][2] + shift_colors[my_type][2] * noise_img[i][j]
			};
			img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}

	return img;
}

cv::Mat genImage2(Config config)
{
	vector<vector<float> > noise_img = genNoise(config);
	vector<vector<int> > noiseType(config.n, vector<int>(config.m));

	float lake_threshold = 0.20;
	float sand_threshold = 0.3;
	float grass1_threshold = 0.47;
	float river_threshold = 0.515;
	float grass2_threshold = 0.72;
	float hill_threshold = 0.89;
	int base_colors[7][3] =
	{
		{ 180, 30, 30 },
		{ 138, 188, 204 },
		{ 60, 245, 25 },
		{ 210, 50, 50 },
		{ 60, 245, 25 },
		{ 19, 180, 110 },
		{ 120, 120, 120 }
	};
	int shift_colors[7][3] =
	{
		{ 30, 0, 0 },
		{ -20, -20, -20 },
		{ -20, -50, -10 },
		{ 10, 0, 0 },
		{ -20, -50, -10 },
		{ 0, -30, 0},
		{ -40, -40, -40 }
	};

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			if (noise_img[i][j] < lake_threshold)
			{
				noiseType[i][j] = 0;
				noise_img[i][j] = (noise_img[i][j] - 0.0) / (lake_threshold - 0.0);
			}
			else if (noise_img[i][j] < sand_threshold)
			{
				noiseType[i][j] = 1;
				noise_img[i][j] = (noise_img[i][j] - lake_threshold) / (sand_threshold - lake_threshold);
			}
			else if (noise_img[i][j] < grass1_threshold)
			{
				noiseType[i][j] = 2;
				noise_img[i][j] = (noise_img[i][j] - sand_threshold) / (grass1_threshold - sand_threshold);
			}
			else if (noise_img[i][j] < river_threshold)
			{
				noiseType[i][j] = 3;
				noise_img[i][j] = (noise_img[i][j] - grass1_threshold) / (river_threshold - grass1_threshold);
			}
			else if (noise_img[i][j] < grass2_threshold)
			{
				noiseType[i][j] = 4;
				noise_img[i][j] = (noise_img[i][j] - river_threshold) / (grass2_threshold - river_threshold);
			}
			else if (noise_img[i][j] < hill_threshold)
			{
				noiseType[i][j] = 5;
				noise_img[i][j] = (noise_img[i][j] - grass2_threshold) / (hill_threshold - grass2_threshold);
			}
			else
			{
				noiseType[i][j] = 6;
				noise_img[i][j] = (noise_img[i][j] - hill_threshold) / (1.0 - hill_threshold);
			}
		}
	}

	cv::Mat img(config.n, config.m, CV_8UC3);

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			int my_type = noiseType[i][j];
			uchar cols[3] =
			{
				base_colors[my_type][0] + shift_colors[my_type][0] * noise_img[i][j],
				base_colors[my_type][1] + shift_colors[my_type][1] * noise_img[i][j],
				base_colors[my_type][2] + shift_colors[my_type][2] * noise_img[i][j]
			};
			img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}

	return img;
}



void test1()
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Config config(512, 512, 1, 75, 6, 0.35 + 0.15 * i, 1.7 + 0.3 * j);

			cv::Mat image = genImage1(config);
			cv::imwrite("test1/img" + to_string(i) + to_string(j) + ".png", image);
		}
	}
}

void test2()
{
	Config config1(512, 512, 2, 75, 6, 0.5, 2.0);
	Config config2(512, 512, 2, 75, 6, 0.5, 2.0);

	cv::Mat image1 = genImage1(config1);
	cv::imwrite("test2/img1.png", image1);

	cv::Mat image2 = genImage2(config2);
	cv::imwrite("test2/img2.png", image2);
}


int main()
{
	//test1();
	test2();

	return 0;
}