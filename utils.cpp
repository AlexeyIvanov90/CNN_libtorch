#include "utils.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <conio.h>
#include <vector>

//make false label csv file
//left arrow - TRUE lable, right arrow - FALSE lable
void examination_error_img(std::string file_csv) {
	std::fstream in(file_csv, std::ios::in);
	std::string line;
	std::string path;
	std::string label;
	std::string result_nn;

	std::vector<std::string> data;

	while (getline(in, line))
		data.push_back(line);

	std::ofstream out;
	out.open("../error/need_delete.csv", std::ios::out);

	for each (auto str in data)
	{
		std::stringstream s(str);
		getline(s, path, ',');
		getline(s, label, ',');
		getline(s, result_nn, ',');

		std::cout << path << std::endl;
		std::cout << "label: " << label << std::endl;
		std::cout << "result_nn: " << result_nn << std::endl;

		auto img = cv::imread(path);

		cv::resize(img, img, cv::Size({ img.cols * 3, img.rows * 3}));

		cv::imshow("<-TRUE LABEL FALSE->", img);
		cv::waitKey(1);

		int key;

		while (true) {
			if (_getch() != 224)
				continue;
			key = _getch();
			if (key == 75 || key == 77)
				break;
		}

		if (key == 75) {
			std::cout << "label true" << std::endl;
		}
		else {
			std::cout << "label false" << std::endl;
			std::cout << "need delete: " << path << std::endl;

			if (out.is_open())
				out << path + "\n";

			cv::imwrite("../false.png", img);
		}
	}

	out.close();
}

