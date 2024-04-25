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
#define big_resl_stats (big_resolution/8)
#define small_resolution 48
#define small_resl_stats (small_resolution/8)

#define max_resolutions 410

vector <vector <vector <float> > > vec_image(NUM_THREADS,
	vector <vector <float> >(max_resolutions, vector <float>(max_resolutions)));
vector <vector <vector <int> > > vec_image_type(NUM_THREADS,
	vector <vector <int> >(max_resolutions, vector <int>(max_resolutions)));

vector <cv::Mat> img_full(NUM_THREADS);
vector <cv::Mat> img_big_res(NUM_THREADS);
vector <cv::Mat> img_small_res(NUM_THREADS);

vector <vector <vector <array<float, 5> > > > big_stats(NUM_THREADS,
	vector <vector <array<float, 5> > >(big_resl_stats, vector <array<float, 5> >(big_resl_stats)));
vector <vector <vector <array<float, 5> > > > small_stats(NUM_THREADS,
	vector <vector <array<float, 5> > >(small_resl_stats, vector <array<float, 5> >(small_resl_stats)));


mt19937 g_rng(4);
uniform_real_distribution<float> g_unif_rng(0.0, 1.0);

float water_threshold = 0.26;
float sand_threshold = 0.42;
float grass_threshold = 0.58;
float rock_threshold = 0.74;
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

	int n, m;

	int seed;

	float scale;
	int octaves;
	float persistance;
	float lacunarity;

	int imSize;

	Config(int id)
	{
		seed = id + g_rng();

		scale = 50.0;
		octaves = 6;
		persistance = 0.4 + g_unif_rng(g_rng) * 0.2;
		lacunarity = 1.75 + g_unif_rng(g_rng) * 0.5;

		n = m = imSize = 50 + g_rng() % 350;
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

void gen_stats(Config config, int thread_id)
{
	auto& big_img_res = img_big_res[thread_id];
	auto& small_img_res = img_small_res[thread_id];

	auto& b_stats = big_stats[thread_id];
	auto& s_stats = small_stats[thread_id];

	array<float, 3> avg_col[5];
	for (int i = 0; i < 5; i++)
	{
		avg_col[i][0] = base_colors[i][0] + shift_colors[i][0] / 2;
		avg_col[i][1] = base_colors[i][1] + shift_colors[i][1] / 2;
		avg_col[i][2] = base_colors[i][2] + shift_colors[i][2] / 2;
	}

	vector <vector <array<int, 5> > > cnt_b(big_resl_stats, vector <array<int, 5> >(big_resl_stats));
	vector <vector <array<int, 5> > > cnt_s(small_resl_stats, vector <array<int, 5> >(small_resl_stats));
	for (auto& it1 : cnt_b) { for (auto& it2 : it1) { for (auto& it3 : it2) { it3 = 0; } } }
	for (auto& it1 : cnt_s) { for (auto& it2 : it1) { for (auto& it3 : it2) { it3 = 0; } } }

	int blk_sz, blk_n;

	blk_sz = big_resolution / big_resl_stats;
	blk_n = big_resl_stats;
	for (int i = 0; i < big_resolution; i++)
	{
		for (int j = 0; j < big_resolution; j++)
		{
			int x = i / blk_sz;
			int y = j / blk_sz;
			
			float dists[5];

			for (int k = 0; k < 5; k++)
			{
				dists[k] = 0;
				for (int l = 0; l < 3; l++)
				{
					dists[k] += abs(avg_col[k][l] - big_img_res.at<cv::Vec3b>(i, j)[l]);
				}
			}

			int min_dist_k = 0;
			for (int k = 1; k < 5; k++)
			{
				if (dists[k] < dists[min_dist_k]) { min_dist_k = k; }
			}

			cnt_b[x][y][min_dist_k]++;
		}
	}

	for (int k = 0; k < 5; k++)
	{
		int mx_val = 1;
		for (int i = 0; i < blk_n; i++)
		{
			for (int j = 0; j < blk_n; j++)
			{
				mx_val = max(mx_val, cnt_b[i][j][k]);
			}
		}

		for (int i = 0; i < blk_n; i++)
		{
			for (int j = 0; j < blk_n; j++)
			{
				b_stats[i][j][k] = float(cnt_b[i][j][k]) / mx_val;
			}
		}
	}

	blk_sz = small_resolution / small_resl_stats;
	blk_n = small_resl_stats;
	for (int i = 0; i < small_resolution; i++)
	{
		for (int j = 0; j < small_resolution; j++)
		{
			int x = i / blk_sz;
			int y = j / blk_sz;

			float dists[5];

			for (int k = 0; k < 5; k++)
			{
				dists[k] = 0;
				for (int l = 0; l < 3; l++)
				{
					dists[k] += abs(avg_col[k][l] - small_img_res.at<cv::Vec3b>(i, j)[l]);
				}
			}

			int min_dist_k = 0;
			for (int k = 1; k < 5; k++)
			{
				if (dists[k] < dists[min_dist_k]) { min_dist_k = k; }
			}

			cnt_s[x][y][min_dist_k]++;
		}
	}

	for (int k = 0; k < 5; k++)
	{
		int mx_val = 1;
		for (int i = 0; i < blk_n; i++)
		{
			for (int j = 0; j < blk_n; j++)
			{
				mx_val = max(mx_val, cnt_s[i][j][k]);
			}
		}

		for (int i = 0; i < blk_n; i++)
		{
			for (int j = 0; j < blk_n; j++)
			{
				s_stats[i][j][k] = float(cnt_s[i][j][k]) / mx_val;
			}
		}
	}
}


int main()
{
	for (int packet_i = 0; packet_i < num_patches; packet_i++)
	{
		float* all_big = new float[big_resolution * big_resolution * 3 * patch_size];
		float* all_b_stats = new float[big_resl_stats * big_resl_stats * 5 * patch_size];
		float* all_small = new float[small_resolution * small_resolution * 3 * patch_size];
		float* all_s_stats = new float[small_resl_stats * small_resl_stats * 5 * patch_size];
		float* all_info = new float[patch_size * 3];

		vector <Config> configs;
		for (int i = 0; i < patch_size; i++) { configs.push_back(Config(packet_i * num_patches + i)); }

		auto start_t = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < patch_size; i++)
		{
			int thread_id = omp_get_thread_num();

			all_info[3 * i] = (configs[i].imSize - 50) / 350.0;
			all_info[3 * i + 1] = (configs[i].persistance - 0.4) / 0.2;
			all_info[3 * i + 2] = (configs[i].lacunarity - 1.75) / 0.5;

			gen_noise(configs[i], thread_id);
			gen_image_and_outputs(configs[i], thread_id);
			gen_stats(configs[i], thread_id);

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

			float* big_stats_p = all_b_stats + big_resl_stats * big_resl_stats * 5 * i;
			for (int ii = 0; ii < big_resl_stats; ii++)
			{
				for (int jj = 0; jj < big_resl_stats; jj++)
				{
					for (int kk = 0; kk < 5; kk++)
					{
						big_stats_p[0] = big_stats[thread_id][ii][jj][kk];
						big_stats_p++;
					}
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

			float* small_stats_p = all_s_stats + small_resl_stats * small_resl_stats * 5 * i;
			for (int ii = 0; ii < small_resl_stats; ii++)
			{
				for (int jj = 0; jj < small_resl_stats; jj++)
				{
					for (int kk = 0; kk < 5; kk++)
					{
						small_stats_p[0] = small_stats[thread_id][ii][jj][kk];
						small_stats_p++;
					}
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

		ofstream file_stats("data/patch" + to_string(packet_i) + "_64x64_stats.bin", ios::binary);
		file_stats.write((char*)all_b_stats, big_resl_stats * big_resl_stats * 5 * patch_size * sizeof(float));
		file_stats.close();

		ofstream file_target("data/patch" + to_string(packet_i) + "_48x48.bin", ios::binary);
		file_target.write((char*)all_small, small_resolution * small_resolution * 3 * patch_size * sizeof(float));
		file_target.close();

		ofstream file_target_stats("data/patch" + to_string(packet_i) + "_48x48_stats.bin", ios::binary);
		file_target_stats.write((char*)all_s_stats, small_resl_stats * small_resl_stats * 5 * patch_size * sizeof(float));
		file_target_stats.close();

		ofstream file_info("data/patch" + to_string(packet_i) + "_info.bin", ios::binary);
		file_info.write((char*)all_info, patch_size * 3 * sizeof(float));
		file_info.close();

		delete[] all_big;
		delete[] all_small;
		delete[] all_info;

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