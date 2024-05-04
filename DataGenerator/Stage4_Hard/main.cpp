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

#define NUM_THREADS 8
#define num_paches 45
#define patch_size 10000

#define max_resolution 512
#define search_resolution 64
#define target_resolution 48

vector <vector <vector <float> > > vec_search(NUM_THREADS,
	vector <vector <float> >(max_resolution, vector <float>(max_resolution)));
vector <vector <vector <float> > > vec_target(NUM_THREADS,
	vector <vector <float> >(max_resolution, vector <float>(max_resolution)));

vector <vector <vector <int> > > vec_search_type(NUM_THREADS,
	vector <vector <int> >(max_resolution, vector <int>(max_resolution)));
vector <vector <vector <int> > > vec_target_type(NUM_THREADS,
	vector <vector <int> >(max_resolution, vector <int>(max_resolution)));

vector <cv::Mat> img_search(NUM_THREADS);
vector <cv::Mat> img_target(NUM_THREADS);

vector <cv::Mat> res_search(NUM_THREADS);
vector <cv::Mat> res_target(NUM_THREADS);
vector <cv::Mat> res_tmp_img(NUM_THREADS);

vector <array<float, 2> > res_output(NUM_THREADS);

mt19937 g_rng(11);
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

	int extraSeed;
	float extraScale;
	int extraOctaves;
	float extraPersistance;
	float extraLacunarity;

	int searchSize;

	float targetExtraNoiseCoef;
	int targetSize;
	int targetLen;
	int target_do_rotate;
	float target_angle;

	Config(int id)
	{
		n = max_resolution;
		m = max_resolution;

		seed = (id + g_rng()) % 1000000000;
		scale = 45.0 + g_unif_rng(g_rng) * 10.0;
		octaves = 5 + g_rng() % 3;
		persistance = 0.37 + g_unif_rng(g_rng) * 0.25;
		lacunarity = 1.75 + g_unif_rng(g_rng) * 0.5;

		extraSeed = id + 1e9;
		extraScale = 20.0 + g_unif_rng(g_rng) * 10.0;
		extraOctaves = 1;
		extraPersistance = 0.37 + g_unif_rng(g_rng) * 0.25;
		extraLacunarity = 1.75 + g_unif_rng(g_rng) * 0.5;

		searchSize = search_resolution;

		targetExtraNoiseCoef = 0.00 + g_unif_rng(g_rng) * 0.25;
		targetSize = target_resolution;
		targetLen = 60 + g_rng() % 140;
		target_do_rotate = g_rng() % 100;
		target_do_rotate = target_do_rotate < 1 ? 0 : 1; // wanted target_do_rotate < 20, but ok...
		if (target_do_rotate) { target_angle = g_unif_rng(g_rng); }
		else { target_angle = 0.0; }
	}
};

void gen_noise(Config config, int thread_id)
{
	auto& search = vec_search[thread_id];
	auto& target = vec_target[thread_id];

	module::Perlin noise;
	module::Perlin extraNoise;

	noise.SetSeed(config.seed);
	noise.SetFrequency(1.0 / config.scale);
	noise.SetOctaveCount(config.octaves);
	noise.SetPersistence(config.persistance);
	noise.SetLacunarity(config.lacunarity);

	extraNoise.SetSeed(config.extraSeed);
	extraNoise.SetFrequency(1.0 / config.extraScale);
	extraNoise.SetOctaveCount(config.extraOctaves);
	extraNoise.SetPersistence(config.extraPersistance);
	extraNoise.SetLacunarity(config.extraLacunarity);

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			search[i][j] = noise.GetValue(i, j, 0) * 0.5 + 0.5;
			target[i][j] = search[i][j] + config.targetExtraNoiseCoef * extraNoise.GetValue(i, j, 0);

			search[i][j] = max(0.0f, min(1.0f, search[i][j]));
			target[i][j] = max(0.0f, min(1.0f, target[i][j]));
		}
	}
}

void gen_high_res_image(Config config, int thread_id)
{
	auto& search = vec_search[thread_id];
	auto& target = vec_target[thread_id];
	auto& search_type = vec_search_type[thread_id];
	auto& target_type = vec_target_type[thread_id];
	auto& search_img = img_search[thread_id];
	auto& target_img = img_target[thread_id];

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

	for (int i = 0; i < config.n; i++)
	{
		for (int j = 0; j < config.m; j++)
		{
			int my_type = search_type[i][j];
			uchar cols[3] =
			{
				base_colors[my_type][0] + shift_colors[my_type][0] * search[i][j],
				base_colors[my_type][1] + shift_colors[my_type][1] * search[i][j],
				base_colors[my_type][2] + shift_colors[my_type][2] * search[i][j]
			};
			search_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);

			my_type = target_type[i][j];
			cols[0] = base_colors[my_type][0] + shift_colors[my_type][0] * target[i][j];
			cols[1] = base_colors[my_type][1] + shift_colors[my_type][1] * target[i][j];
			cols[2] = base_colors[my_type][2] + shift_colors[my_type][2] * target[i][j];
			target_img.at<cv::Vec3b>(i, j) = cv::Vec3b(cols[0], cols[1], cols[2]);
		}
	}
}

void gen_res_images(Config config, int thread_id)
{
	auto& searchFull = img_search[thread_id];
	auto& targetFull = img_target[thread_id];
	auto& tmp_img = res_tmp_img[thread_id];

	auto res_search_size = cv::Size(config.searchSize, config.searchSize);
	cv::resize(searchFull, res_search[thread_id], res_search_size, 0, 0, cv::INTER_LINEAR);

	mt19937 rng(config.seed);

	if (config.target_do_rotate == 0)
	{
		float cx, cy;

		int rectFullSize = config.targetLen;
		int x = rng() % (config.n - rectFullSize);
		int y = rng() % (config.m - rectFullSize);

		cv::Rect rect(x, y, rectFullSize, rectFullSize);
		tmp_img = targetFull(rect);

		auto res_target_size = cv::Size(config.targetSize, config.targetSize);
		cv::resize(tmp_img, res_target[thread_id], res_target_size, 0, 0, cv::INTER_LINEAR);

		cx = (float(x) + rectFullSize / 2.0) / config.n;
		cy = (float(y) + rectFullSize / 2.0) / config.m;

		res_output[thread_id][0] = cx;
		res_output[thread_id][1] = cy;
	}
	else
	{
		int rectFullSize = config.targetLen;
		float rotationDegrees = config.target_angle * 360.0;
		float pi = 3.14159265359;
		float sq2 = 1.41421356237;
		float ang_r = rotationDegrees * pi / 180.0;

		float maxCoef = sq2 * max(abs(sin(ang_r + pi / 4)), abs(cos(ang_r + pi / 4)));
		int border = rectFullSize / 2.0 * maxCoef + 1.5;

		int cent_x = border + rng() % (config.n - 2 * border);
		int cent_y = border + rng() % (config.m - 2 * border);

		cv::Point2f center(cent_x, cent_y);
		cv::Size2f rectSize(rectFullSize, rectFullSize);

		tmp_img = cv::getRotationMatrix2D(center, rotationDegrees, maxCoef);

		cv::Mat tmp_img2, tmp_img3;
		cv::warpAffine(targetFull, tmp_img2, tmp_img, targetFull.size());

		cv::RotatedRect rotatedRect(center, rectSize, rotationDegrees);
		cv::Rect boundingRect = rotatedRect.boundingRect();

		tmp_img3 = tmp_img2(boundingRect);

		auto res_target_size = cv::Size(config.targetSize, config.targetSize);
		cv::resize(tmp_img3, res_target[thread_id], res_target_size, 0, 0, cv::INTER_LINEAR);

		float cx, cy;

		cx = float(cent_x) / config.n;
		cy = float(cent_y) / config.m;

		res_output[thread_id][0] = cx;
		res_output[thread_id][1] = cy;
	}
}


int main()
{
	for (int i = 0; i < NUM_THREADS; i++)
	{
		img_search[i] = cv::Mat(max_resolution, max_resolution, CV_8UC3);
		img_target[i] = cv::Mat(max_resolution, max_resolution, CV_8UC3);
		res_search[i] = cv::Mat(search_resolution, search_resolution, CV_8UC3);
		res_target[i] = cv::Mat(target_resolution, target_resolution, CV_8UC3);
	}

	for (int packet_i = 0; packet_i < num_paches; packet_i++)
	{
		float* all_search = new float[search_resolution * search_resolution * 3 * patch_size];
		float* all_target = new float[target_resolution * target_resolution * 3 * patch_size];
		float* all_output = new float[2 * patch_size];

		vector <Config> configs;
		for (int i = 0; i < patch_size; i++) { configs.push_back(Config(packet_i * num_paches + i)); }

		auto start_t = chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(NUM_THREADS)
		for (int i = 0; i < patch_size; i++)
		{
			int thread_id = omp_get_thread_num();
			gen_noise(configs[i], thread_id);
			gen_high_res_image(configs[i], thread_id);
			gen_res_images(configs[i], thread_id);

			float* search_p = all_search + search_resolution * search_resolution * 3 * i;
			for (int ii = 0; ii < search_resolution; ii++)
			{
				for (int jj = 0; jj < search_resolution; jj++)
				{
					search_p[0] = res_search[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					search_p[1] = res_search[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					search_p[2] = res_search[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					search_p += 3;
				}
			}

			float* target_p = all_target + target_resolution * target_resolution * 3 * i;
			for (int ii = 0; ii < target_resolution; ii++)
			{
				for (int jj = 0; jj < target_resolution; jj++)
				{
					target_p[0] = res_target[thread_id].at<cv::Vec3b>(ii, jj)[2] / 255.0;
					target_p[1] = res_target[thread_id].at<cv::Vec3b>(ii, jj)[1] / 255.0;
					target_p[2] = res_target[thread_id].at<cv::Vec3b>(ii, jj)[0] / 255.0;
					target_p += 3;
				}
			}

			float* output_p = all_output + 2 * i;
			output_p[0] = res_output[thread_id][0];
			output_p[1] = res_output[thread_id][1];

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
		
		ofstream file_output("data/patch" + to_string(packet_i) + "_output.bin", ios::binary);
		file_output.write((char*)all_output, 2 * patch_size * sizeof(float));
		file_output.close();

		delete[] all_search;
		delete[] all_target;
		delete[] all_output;

		/*
		for (int ti = 0; ti < NUM_THREADS; ti++)
		{
			cv::imwrite("images/" + to_string(ti) + "_search.png", res_search[ti]);
			cv::imwrite("images/" + to_string(ti) + "_target.png", res_target[ti]);
		}
		//*/
	}

	return 0;
}