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

template<typename T>
using vec = std::vector<T>;

#define NUM_THREADS 8
#define num_patches 11
#define patch_size 10000

#define search_resl 64
#define target_resl 48

#define full_resl 512
#define part_resl 96

vec <vec <vec <float> > > noises(NUM_THREADS, vec <vec <float> >(full_resl, vec <float>(full_resl)));
vec <vec <vec <int> > > noises_type(NUM_THREADS, vec <vec <int> >(full_resl, vec <int>(full_resl)));
vec <vec <vec <int> > > pref_sums(NUM_THREADS, vec <vec <int> >(full_resl + 1, vec <int>(full_resl + 1, 0)));
vec <pair<int, int> > targets_xy(NUM_THREADS);
vec <vec <vec <float> > > masks(NUM_THREADS, vec <vec <float> >(search_resl, vec <float>(search_resl)));

vec <cv::Mat> imgs_full(NUM_THREADS);
vec <cv::Mat> imgs_search(NUM_THREADS);
vec <cv::Mat> imgs_target(NUM_THREADS);
vec <cv::Mat> imgs_tmp(NUM_THREADS);

mt19937 g_rng(9);
uniform_real_distribution<float> g_unif_rng(0.0, 1.0);

float water_threshold = 0.24;
float sand_threshold = 0.42;
float grass_threshold = 0.58;
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

class Config
{
public:

	int seed;

	float scale;
	int octaves;
	float persistance;
	float lacunarity;

	int target_type;

	Config(int id)
	{
		seed = (id + g_rng()) % 1000000000;

		scale = 50.0;
		octaves = 6;
		persistance = 0.4 + g_unif_rng(g_rng) * 0.175;
		lacunarity = 1.75 + g_unif_rng(g_rng) * 0.45;

		target_type = g_rng() % 5;
	}
};

void gen_noise(Config config, int thread_id)
{
	auto& _noise = noises[thread_id];
	auto& noise_type = noises_type[thread_id];
	auto& pref_sum = pref_sums[thread_id];
	auto& target_xy = targets_xy[thread_id];

	module::Perlin noise_g;

	noise_g.SetSeed(config.seed);
	noise_g.SetFrequency(1.0 / config.scale);
	noise_g.SetOctaveCount(config.octaves);
	noise_g.SetPersistence(config.persistance);
	noise_g.SetLacunarity(config.lacunarity);

	for (int i = 0; i < full_resl; i++)
	{
		for (int j = 0; j < full_resl; j++)
		{
			_noise[i][j] = noise_g.GetValue(i, j, 0) * 0.5 + 0.5;
			_noise[i][j] = min(1.0f, max(0.0f, _noise[i][j]));
		}
	}
	for (int i = 0; i < full_resl; i++)
	{
		for (int j = 0; j < full_resl; j++)
		{
			if (_noise[i][j] < water_threshold)
			{
				noise_type[i][j] = 0;
				_noise[i][j] = (_noise[i][j] - 0.0) / (water_threshold - 0.0);
			}
			else if (_noise[i][j] < sand_threshold)
			{
				noise_type[i][j] = 1;
				_noise[i][j] = (_noise[i][j] - water_threshold) / (sand_threshold - water_threshold);
			}
			else if (_noise[i][j] < grass_threshold)
			{
				noise_type[i][j] = 2;
				_noise[i][j] = (_noise[i][j] - sand_threshold) / (grass_threshold - sand_threshold);
			}
			else if (_noise[i][j] < rock_threshold)
			{
				noise_type[i][j] = 3;
				_noise[i][j] = (_noise[i][j] - grass_threshold) / (rock_threshold - grass_threshold);
			}
			else
			{
				noise_type[i][j] = 4;
				_noise[i][j] = (_noise[i][j] - rock_threshold) / (1.0 - rock_threshold);
			}
		}
	}

	for (int i = 0; i < full_resl; i++)
	{
		for (int j = 0; j < full_resl; j++)
		{
			pref_sum[i + 1][j + 1] = noise_type[i][j] == config.target_type;
			pref_sum[i + 1][j + 1] += pref_sum[i][j + 1] + pref_sum[i + 1][j] - pref_sum[i][j];
		}
	}

	vector <pair<int, int> > good_xy;

	int border = 24;
	int sum_pxl = (part_resl - 2 * border) * (part_resl - 2 * border);
	for (int i = 0; i + part_resl <= full_resl; i++)
	{
		for (int j = 0; j + part_resl <= full_resl; j++)
		{
			int x1 = i + border, y1 = j + border, x2 = i + part_resl - border, y2 = j + part_resl - border;
			int cnt = pref_sum[x2][y2] - pref_sum[x1][y2] - pref_sum[x2][y1] + pref_sum[x1][y1];
			if (cnt >= sum_pxl * 0.65 && cnt <= sum_pxl * 0.85)
			{
				good_xy.push_back({ i, j });
			}
		}
	}

	mt19937 rng(config.seed);

	if (good_xy.empty())
	{
		config.seed = rng() % 1000000000;
		gen_noise(config, thread_id);
		return;
	}

	target_xy = good_xy[rng() % good_xy.size()];
}

void gen_images(Config config, int thread_id)
{
	auto& _noise = noises[thread_id];
	auto& noise_type = noises_type[thread_id];
	auto& full_img = imgs_full[thread_id];
	auto& search = imgs_search[thread_id];
	auto& target = imgs_target[thread_id];
	auto& tmp_img = imgs_tmp[thread_id];
	auto& target_xy = targets_xy[thread_id];
	auto& mask = masks[thread_id];

	for (int i = 0; i < full_resl; i++)
	{
		for (int j = 0; j < full_resl; j++)
		{
			int my_type = noise_type[i][j];
			uchar cols[3] =
			{
				cols[0] = base_colors[my_type][0] + shift_colors[my_type][0] * _noise[i][j],
				cols[1] = base_colors[my_type][1] + shift_colors[my_type][1] * _noise[i][j],
				cols[2] = base_colors[my_type][2] + shift_colors[my_type][2] * _noise[i][j]
			};
			full_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}
	auto size = cv::Size(search_resl, search_resl);
	cv::resize(full_img, search, size, 0, 0, cv::INTER_LINEAR);
	
	for (int i = 0; i < part_resl; i++)
	{
		for (int j = 0; j < part_resl; j++)
		{
			int x = target_xy.first + i, y = target_xy.second + j;
			tmp_img.at<cv::Vec3b>(i, j) = full_img.at<cv::Vec3b>(x, y);
		}
	}
	size = cv::Size(target_resl, target_resl);
	cv::resize(tmp_img, target, size, 0, 0, cv::INTER_LINEAR);

	int blk_sz = full_resl / search_resl;
	for (int i = 0; i < search_resl; i++)
	{
		for (int j = 0; j < search_resl; j++)
		{
			int x = i * blk_sz, y = j * blk_sz;
			int cnt = 0;
			for (int ii = 0; ii < blk_sz; ii++)
			{
				for (int jj = 0; jj < blk_sz; jj++)
				{
					if (noise_type[x + ii][y + jj] == config.target_type)
					{
						cnt++;
					}
				}
			}
			mask[i][j] = float(cnt) / (blk_sz * blk_sz);
		}
	}
}


int main()
{
	for (int i = 0; i < NUM_THREADS; i++)
	{
		imgs_full[i] = cv::Mat(full_resl, full_resl, CV_8UC3);
		imgs_search[i] = cv::Mat(search_resl, search_resl, CV_8UC3);
		imgs_target[i] = cv::Mat(target_resl, target_resl, CV_8UC3);
		imgs_tmp[i] = cv::Mat(part_resl, part_resl, CV_8UC3);
	}

	for (int packet_i = 0; packet_i < num_patches; packet_i++)
	{
		float* all_search = new float[search_resl * search_resl * 3 * patch_size];
		float* all_target = new float[target_resl * target_resl * 3 * patch_size];
		float* all_mask = new float[search_resl * search_resl * patch_size];

		vector <Config> configs;
		for (int i = 0; i < patch_size; i++) { configs.push_back(Config(packet_i * num_patches + i)); }

		auto start_t = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < patch_size; i++)
		{
			int thread_id = omp_get_thread_num();

			gen_noise(configs[i], thread_id);
			gen_images(configs[i], thread_id);

			float* search_p = all_search + search_resl * search_resl * 3 * i;
			for (int ii = 0; ii < search_resl; ii++)
			{
				for (int jj = 0; jj < search_resl; jj++)
				{
					search_p[0] = imgs_search[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					search_p[1] = imgs_search[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					search_p[2] = imgs_search[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					search_p += 3;
				}
			}
			
			float* target_p = all_target + target_resl * target_resl * 3 * i;
			for (int ii = 0; ii < target_resl; ii++)
			{
				for (int jj = 0; jj < target_resl; jj++)
				{
					target_p[0] = imgs_target[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					target_p[1] = imgs_target[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					target_p[2] = imgs_target[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					target_p += 3;
				}
			}
			
			float* mask_p = all_mask + search_resl * search_resl * i;
			for (int ii = 0; ii < search_resl; ii++)
			{
				for (int jj = 0; jj < search_resl; jj++)
				{
					mask_p[0] = masks[thread_id][ii][jj];
					mask_p++;
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
		file_search.write((char*)all_search, search_resl * search_resl * 3 * patch_size * sizeof(float));
		file_search.close();

		ofstream file_stats("data/patch" + to_string(packet_i) + "_target.bin", ios::binary);
		file_stats.write((char*)all_target, target_resl * target_resl * 3 * patch_size * sizeof(float));
		file_stats.close();

		ofstream file_target("data/patch" + to_string(packet_i) + "_mask.bin", ios::binary);
		file_target.write((char*)all_mask, search_resl * search_resl * patch_size * sizeof(float));
		file_target.close();

		delete[] all_search;
		delete[] all_target;
		delete[] all_mask;

		/*
		for (int ti = 0; ti < NUM_THREADS; ti++)
		{
			cv::imwrite("images/" + to_string(ti) + "_search.png", imgs_search[ti]);
			cv::imwrite("images/" + to_string(ti) + "_target.png", imgs_target[ti]);

			cv::Mat mask_img(search_resl, search_resl, CV_32F);
			for (int i = 0; i < search_resl; ++i)
			{
				for (int j = 0; j < search_resl; ++j)
				{
					mask_img.at<float>(i, j) = masks[ti][i][j];
				}
			}
			mask_img.convertTo(mask_img, CV_8U, 255.0);
			cv::imwrite("images/" + to_string(ti) + "_mask.png", mask_img);
		}
		//*/
	}

	return 0;
}