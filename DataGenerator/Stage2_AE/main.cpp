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
#define num_patches 21
#define patch_size 10000

#define big_resolution 64
#define small_resolution 48

#define max_resolutions 410

vector <vector <vector <float> > > vec_image(NUM_THREADS,
	vector <vector <float> >(max_resolutions, vector <float>(max_resolutions)));
vector <vector <vector <int> > > vec_image_type(NUM_THREADS,
	vector <vector <int> >(max_resolutions, vector <int>(max_resolutions)));

vector <cv::Mat> img_full(NUM_THREADS);
vector <cv::Mat> img_big_res(NUM_THREADS);
vector <cv::Mat> img_small_res(NUM_THREADS);

mt19937 g_rng(4);
uniform_real_distribution<float> g_unif_rng(0.0, 1.0);

class Config
{
public:

	int n, m;

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

		n = m = tSize = 50 + g_rng() % 350;
	}
};

void gen_noise(Config config, int thread_id)
{
	auto& image = vec_image[thread_id];

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
			image[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			image[i][j] = min(1.0f, max(0.0f, image[i][j]));
		}
	}
}

void gen_image_and_outputs(Config config, int thread_id)
{
	auto& image = vec_image[thread_id];
	auto& image_type = vec_image_type[thread_id];
	img_full[thread_id] = cv::Mat(config.n, config.m, CV_8UC3);
	auto& full_img = img_full[thread_id];
	img_big_res[thread_id] = cv::Mat(big_resolution, big_resolution, CV_8UC3);
	auto& big_img_res = img_big_res[thread_id];
	img_small_res[thread_id] = cv::Mat(small_resolution, small_resolution, CV_8UC3);
	auto& small_img_res = img_small_res[thread_id];

	float water_threshold = 0.3;
	float sand_threshold = 0.44;
	float grass_threshold = 0.56;
	float rock_threshold = 0.7;
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
			if (image[i][j] < water_threshold)
			{
				image_type[i][j] = 0;
				image[i][j] -= 0.0;
				image[i][j] /= (water_threshold - 0.0);
			}
			else if (image[i][j] < sand_threshold)
			{
				image_type[i][j] = 1;
				image[i][j] -= water_threshold;
				image[i][j] /= (sand_threshold - water_threshold);
			}
			else if (image[i][j] < grass_threshold)
			{
				image_type[i][j] = 2;
				image[i][j] -= sand_threshold;
				image[i][j] /= (grass_threshold - sand_threshold);
			}
			else if (image[i][j] < rock_threshold)
			{
				image_type[i][j] = 3;
				image[i][j] -= grass_threshold;
				image[i][j] /= (rock_threshold - grass_threshold);
			}
			else
			{
				image_type[i][j] = 4;
				image[i][j] -= rock_threshold;
				image[i][j] /= (1.0 - rock_threshold);
			}
		}
	}
	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			int my_type = image_type[i][j];
			uchar cols[3] =
			{
				cols[0] = base_colors[my_type][0] + shift_colors[my_type][0] * image[i][j],
				cols[1] = base_colors[my_type][1] + shift_colors[my_type][1] * image[i][j],
				cols[2] = base_colors[my_type][2] + shift_colors[my_type][2] * image[i][j]
			};
			full_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}
	auto size = cv::Size(big_resolution, big_resolution);
	cv::resize(full_img, big_img_res, size, 0, 0, cv::INTER_LINEAR);
	size = cv::Size(small_resolution, small_resolution);
	cv::resize(full_img, small_img_res, size, 0, 0, cv::INTER_LINEAR);
}


int main()
{
	for (int packet_i = 0; packet_i < num_patches; packet_i++)
	{
		float* all_big = new float[big_resolution * big_resolution * 3 * patch_size];
		float* all_small = new float[small_resolution * small_resolution * 3 * patch_size];

		vector <Config> configs;
		for (int i = 0; i < patch_size; i++) { configs.push_back(Config(packet_i * num_patches + i)); }

		auto start_t = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < patch_size; i++)
		{
			int thread_id = omp_get_thread_num();
			gen_noise(configs[i], thread_id);
			gen_image_and_outputs(configs[i], thread_id);

			float* big_p = all_big + big_resolution * big_resolution * 3 * i;
			for (int ii = 0; ii < big_resolution; ii++)
			{
				for (int jj = 0; jj < big_resolution; jj++)
				{
					big_p[0] = img_big_res[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					big_p[1] = img_big_res[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					big_p[2] = img_big_res[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					big_p += 3;
				}
			}

			float* small_p = all_small + small_resolution * small_resolution * 3 * i;
			for (int ii = 0; ii < small_resolution; ii++)
			{
				for (int jj = 0; jj < small_resolution; jj++)
				{
					small_p[0] = img_small_res[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					small_p[1] = img_small_res[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					small_p[2] = img_small_res[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					small_p += 3;
				}
			}

			if (thread_id == 0 && i % 50 == 0)
			{
				auto end_t = chrono::high_resolution_clock::now();
				chrono::duration<double> elapsed_t = end_t - start_t;
				cout << "packet_i: " << packet_i << " i: " << i << "  " << elapsed_t.count() << "s\n";
			}
		}

		ofstream file_search("data/patch" + to_string(packet_i) + "_64x64.bin", ios::binary);
		file_search.write((char*)all_big, big_resolution * big_resolution * 3 * patch_size * sizeof(float));
		file_search.close();

		ofstream file_target("data/patch" + to_string(packet_i) + "_48x48.bin", ios::binary);
		file_target.write((char*)all_small, small_resolution * small_resolution * 3 * patch_size * sizeof(float));
		file_target.close();

		delete[] all_big;
		delete[] all_small;

		/*
		for (int i = 0; i < NUM_THREADS; i++)
		{
			cv::imwrite("images/64x64_" + to_string(i) + ".png", img_big_res[i]);
			cv::imwrite("images/48x48_" + to_string(i) + ".png", img_small_res[i]);
		}
		//*/
	}

	return 0;
}