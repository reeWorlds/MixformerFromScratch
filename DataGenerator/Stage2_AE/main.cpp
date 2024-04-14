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

#define NUM_THREADS 8
//#define num_patches 21
#define num_patches 1
//#define patch_size 10000
#define patch_size 8

#define search_resolution 64
#define target_resolution 48

vector <vector <vector <float> > > vec_search(NUM_THREADS);
vector <vector <vector <int> > > vec_search_type(NUM_THREADS);
vector <vector <vector <float> > > vec_target(NUM_THREADS);
vector <vector <vector <int> > > vec_target_type(NUM_THREADS);

vector <cv::Mat> img_search(NUM_THREADS);
vector <cv::Mat> img_search_res(NUM_THREADS);
vector <cv::Mat> img_target(NUM_THREADS);
vector <cv::Mat> img_target_res(NUM_THREADS);

mt19937 g_rng(3);
uniform_real_distribution<float> g_unif_rng(0.0, 1.0);

class Config
{
public:

	int sn, sm, tn, tm;

	int seed;

	float scale;
	int octaves;
	float persistance;
	float lacunarity;

	int sSize;
	int tSize;

	Config(int id)
	{
		seed = id + g_rng();

		scale = 50.0;
		octaves = 6;
		persistance = 0.5;
		lacunarity = 2.0;

		sn = sm = sSize = 300 + g_rng() % 250;
		tn = tm = tSize = 50 + g_rng() % 150;
	}
};

void gen_noise(Config config, int thread_id)
{
	vec_search[thread_id] = vector <vector <float> >(config.sn, vector <float>(config.sm));
	auto& search = vec_search[thread_id];
	vec_target[thread_id] = vector <vector <float> >(config.tn, vector <float>(config.tm));
	auto& target = vec_target[thread_id];

	module::Perlin noise;

	noise.SetSeed(config.seed);
	noise.SetFrequency(1.0 / config.scale);
	noise.SetOctaveCount(config.octaves);
	noise.SetPersistence(config.persistance);
	noise.SetLacunarity(config.lacunarity);

	for (int i = 0; i < config.sn; i++)
	{
		for (int j = 0; j < config.sm; j++)
		{
			search[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			search[i][j] = min(1.0f, max(0.0f, search[i][j]));
		}
	}

	for (int i = 0; i < config.tn; i++)
	{
		for (int j = 0; j < config.tm; j++)
		{
			target[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			target[i][j] = min(1.0f, max(0.0f, target[i][j]));
		}
	}
}

void gen_image_and_outputs(Config config, int thread_id)
{
	auto& search = vec_search[thread_id];
	vec_search_type[thread_id] = vector <vector <int> >(config.sn, vector <int>(config.sm));
	auto& search_type = vec_search_type[thread_id];
	img_search[thread_id] = cv::Mat(config.sn, config.sm, CV_8UC3);
	auto& search_img = img_search[thread_id];
	img_search_res[thread_id] = cv::Mat(search_resolution, search_resolution, CV_8UC3);
	auto& search_img_res = img_search_res[thread_id];

	auto& target = vec_target[thread_id];
	vec_target_type[thread_id] = vector <vector <int> >(config.tn, vector <int>(config.tm));
	auto& target_type = vec_target_type[thread_id];
	img_target[thread_id] = cv::Mat(config.tn, config.tm, CV_8UC3);
	auto& target_img = img_target[thread_id];
	img_target_res[thread_id] = cv::Mat(target_resolution, target_resolution, CV_8UC3);
	auto& target_img_res = img_target_res[thread_id];

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

	for (int i = 0; i < config.sn; i++)
	{
		for (int j = 0; j < config.sm; j++)
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
	for (int i = 0; i < config.sn; i++)
	{
		for (int j = 0; j < config.sm; j++)
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
	auto size = cv::Size(search_resolution, search_resolution);
	cv::resize(search_img, search_img_res, size, 0, 0, cv::INTER_LINEAR);

	for (int i = 0; i < config.tn; i++)
	{
		for (int j = 0; j < config.tm; j++)
		{
			if (target[i][j] < water_threshold)
			{
				target_type[i][j] = 0;
				target[i][j] -= 0.0;
				target[i][j] /= (water_threshold - 0.0);
			}
			else if (target[i][j] < sand_threshold)
			{
				target_type[i][j] = 1;
				target[i][j] -= water_threshold;
				target[i][j] /= (sand_threshold - water_threshold);
			}
			else if (target[i][j] < grass_threshold)
			{
				target_type[i][j] = 2;
				target[i][j] -= sand_threshold;
				target[i][j] /= (grass_threshold - sand_threshold);
			}
			else if (target[i][j] < rock_threshold)
			{
				target_type[i][j] = 3;
				target[i][j] -= grass_threshold;
				target[i][j] /= (rock_threshold - grass_threshold);
			}
			else
			{
				target_type[i][j] = 4;
				target[i][j] -= rock_threshold;
				target[i][j] /= (1.0 - rock_threshold);
			}
		}
	}
	for (int i = 0; i < config.tn; i++)
	{
		for (int j = 0; j < config.tm; j++)
		{
			int my_type = target_type[i][j];
			uchar cols[3] =
			{
				cols[0] = base_colors[my_type][0] + shift_colors[my_type][0] * target[i][j],
				cols[1] = base_colors[my_type][1] + shift_colors[my_type][1] * target[i][j],
				cols[2] = base_colors[my_type][2] + shift_colors[my_type][2] * target[i][j]
			};
			target_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}
	size = cv::Size(target_resolution, target_resolution);
	cv::resize(target_img, target_img_res, size, 0, 0, cv::INTER_LINEAR);
}



int main()
{
	for (int packet_i = 0; packet_i < num_patches; packet_i++)
	{
		float* all_search = new float[search_resolution * search_resolution * 3 * patch_size];
		float* all_target = new float[target_resolution * target_resolution * 3 * patch_size];

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

			float* target_p = all_target + target_resolution * target_resolution * 3 * i;
			for (int ii = 0; ii < target_resolution; ii++)
			{
				for (int jj = 0; jj < target_resolution; jj++)
				{
					target_p[0] = img_target_res[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					target_p[1] = img_target_res[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					target_p[2] = img_target_res[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					target_p += 3;
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

		ofstream file_target("data/patch" + to_string(packet_i) + "_target.bin", ios::binary);
		file_target.write((char*)all_target, target_resolution * target_resolution * 3 * patch_size
			* sizeof(float));
		file_target.close();

		delete[] all_search;
		delete[] all_target;

		for (int i = 0; i < NUM_THREADS; i++)
		{
			cv::imwrite("images/search" + to_string(i) + ".png", img_search_res[i]);
			cv::imwrite("images/target" + to_string(i) + ".png", img_target_res[i]);
		}
	}

	return 0;
}