#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <array>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <noise.h>
using namespace std;
using namespace noise;

#define NUM_THREADS 4
#define num_patches 11
#define patch_size 10000

#define max_resolution 256
#define search_resolution 64

vector <vector <vector <float> > > vec_search(NUM_THREADS,
	vector <vector <float> >(max_resolution, vector <float>(max_resolution)));

vector <vector <vector <int> > > vec_search_type(NUM_THREADS,
	vector <vector <int> >(max_resolution, vector <int>(max_resolution)));

vector <cv::Mat> img_search(NUM_THREADS);
vector <cv::Mat> img_search_res(NUM_THREADS);

vector <uint8_t> vec_inp_class(NUM_THREADS);

vector <cv::Mat> img_output(NUM_THREADS);

mt19937 g_rng(2);
uniform_real_distribution<float> g_unif_rng(0.0, 1.0);

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

	int searchSize;
	int searchLen;

	Config(int id)
	{
		n = max_resolution;
		m = max_resolution;

		seed = id + g_rng();
		scale = 50.0;
		octaves = 6;
		persistance = 0.5;
		lacunarity = 2.0;

		searchSize = search_resolution;
		searchLen = 128;
	}
};

void gen_noise(Config config, int thread_id)
{
	auto& search = vec_search[thread_id];

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
			search[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			search[i][j] = min(1.0f, max(0.0f, search[i][j]));
		}
	}
}

void gen_image_and_outputs(Config config, int thread_id)
{
	auto& search = vec_search[thread_id];
	auto& search_type = vec_search_type[thread_id];
	auto& search_img = img_search[thread_id];
	auto& search_img_res = img_search_res[thread_id];
	auto& inp_class = vec_inp_class[thread_id];
	auto& output = img_output[thread_id];

	float water_threshold = 0.25;
	float sand_threshold = 0.4;
	float grass_threshold = 0.6;
	float rock_threshold = 0.8;
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
			if (search[i][j] < water_threshold)
			{
				search_type[i][j] = 0;
				search[i][j] -= 0.0;
				search[i][j] /= (water_threshold - 0.0);
			}
			else if (search[i][j] < sand_threshold)
			{
				search_type[i][j] = 1;
				search[i][j] -= water_threshold;
				search[i][j] /= (sand_threshold - water_threshold);
			}
			else if (search[i][j] < grass_threshold)
			{
				search_type[i][j] = 2;
				search[i][j] -= sand_threshold;
				search[i][j] /= (grass_threshold - sand_threshold);
			}
			else if (search[i][j] < rock_threshold)
			{
				search_type[i][j] = 3;
				search[i][j] -= grass_threshold;
				search[i][j] /= (rock_threshold - grass_threshold);
			}
			else
			{
				search_type[i][j] = 4;
				search[i][j] -= rock_threshold;
				search[i][j] /= (1.0 - rock_threshold);
			}
		}
	}

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			int my_type = search_type[i][j];
			uchar cols[3] =
			{
				cols[0] = base_colors[my_type][0] + shift_colors[my_type][0] * search[i][j],
				cols[1] = base_colors[my_type][1] + shift_colors[my_type][1] * search[i][j],
				cols[2] = base_colors[my_type][2] + shift_colors[my_type][2] * search[i][j]
			};
			search_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}

	auto size = cv::Size(config.searchSize, config.searchSize);
	cv::resize(search_img, search_img_res, size, 0, 0, cv::INTER_LINEAR);

	mt19937 rng(config.seed * config.seed);

	inp_class = rng() % 5;

	int blockSize = max_resolution / search_resolution;
	for (int i = 0; i < config.searchSize; i++)
	{
		for (int j = 0; j < config.searchSize; j++)
		{
			int cnt = 0;
			for (int ii = 0; ii < blockSize; ii++)
			{
				for (int jj = 0; jj < blockSize; jj++)
				{
					int x = i * blockSize + ii;
					int y = j * blockSize + jj;

					if (search_type[x][y] == inp_class)
					{
						cnt++;
					}
				}
			}
			float val = cnt / (float)(blockSize * blockSize);
			int val_i = (int)(val * 255);

			output.at<uchar>(i, j) = val_i;
		}
	}
}



int main()
{
	for (int i = 0; i < NUM_THREADS; i++)
	{
		img_search[i] = cv::Mat(max_resolution, max_resolution, CV_8UC3);
		img_search_res[i] = cv::Mat(search_resolution, search_resolution, CV_8UC3);
		img_output[i] = cv::Mat(search_resolution, search_resolution, CV_8UC1);
	}

	for (int packet_i = 0; packet_i < num_patches; packet_i++)
	{
		float* all_search = new float[search_resolution * search_resolution * 3 * patch_size];
		uint8_t* all_class = new uint8_t[patch_size];
		uint8_t* all_output = new uint8_t[search_resolution * search_resolution * patch_size];

		vector <Config> configs;
		for (int i = 0; i < patch_size; i++) { configs.push_back(Config(packet_i * num_patches + i)); }

		auto start_t = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < patch_size; i++)
		{
			int thread_id = omp_get_thread_num();
			gen_noise(configs[i], thread_id);
			gen_image_and_outputs(configs[i], thread_id);

			float* search_p = all_search + search_resolution * search_resolution * 3 * i;
			for (int ii = 0; ii < search_resolution; ii++)
			{
				for (int jj = 0; jj < search_resolution; jj++)
				{
					search_p[0] = img_search_res[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					search_p[1] = img_search_res[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					search_p[2] = img_search_res[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					search_p += 3;
				}
			}

			all_class[i] = vec_inp_class[thread_id];

			uint8_t* output_p = all_output + search_resolution * search_resolution * i;
			for (int ii = 0; ii < search_resolution; ii++)
			{
				for (int jj = 0; jj < search_resolution; jj++)
				{
					output_p[0] = img_output[thread_id].at<uchar>(ii, jj);
					search_p++;
				}
			}

			if (thread_id == 0 && i % 50 == 0)
			{
				auto end_t = chrono::high_resolution_clock::now();
				chrono::duration<double> elapsed_t = end_t - start_t;
				cout << "packet_i: " << packet_i << " i: " << i << "  " << elapsed_t.count() << "s\n";
			}
		}

		ofstream file_search("data/patch" + to_string(packet_i) + "_search.bin", ios::binary);
		file_search.write((char*)all_search, search_resolution * search_resolution * 3 * patch_size
			* sizeof(float));
		file_search.close();
		
		ofstream file_class("data/patch" + to_string(packet_i) + "_class.bin", ios::binary);
		file_class.write((char*)all_class, patch_size * sizeof(uint8_t));
		file_class.close();
		
		ofstream file_output("data/patch" + to_string(packet_i) + "_output.bin", ios::binary);
		file_output.write((char*)all_output, search_resolution * search_resolution * patch_size
			* sizeof(uint8_t));
		file_output.close();

		delete[] all_search;
		delete[] all_class;
		delete[] all_output;

		//for (int i = 0; i < NUM_THREADS; i++)
		//{
		//	cv::imwrite("images/input" + to_string(i) + ".png", img_search_res[i]);
		//	cv::imwrite("images/output" + to_string(i) + ".png", img_output[i]);
		//}
	}

	return 0;
}