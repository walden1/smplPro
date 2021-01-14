

#include "smpl_model.h"

using namespace std;

#ifdef _WIN32
#define WINPAUSE system("pause")
#endif

bool read_data(string file, vector<float> &dat, int size)
{
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return false;
	}
	dat.clear();
	dat.resize(size);
	std::copy(std::istream_iterator<float>(inf), std::istream_iterator<float>(), dat.data());
	inf.close();
	return true;
}

bool read_betas(string path, string fname, vector<float> &betas)
{
	string file = path + fname + ".betas";
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return false;
	}
	betas.clear();
	betas.resize(10);
	std::copy(std::istream_iterator<float>(inf), std::istream_iterator<float>(), betas.data());
	inf.close();
	return true;
}

bool read_poses(string path, string fname, vector<vector<float>> &poses)
{
	string file = path + fname + ".poses";
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return false;
	}

	poses.clear();
	string line;
	stringstream lineInput;
	std::vector<float> numbers;
	numbers.resize(72);
	while (getline(inf, line))
	{
		lineInput.str("");
		lineInput.clear();
		lineInput << line;
		numbers.clear();
		copy(std::istream_iterator<float>(lineInput), std::istream_iterator<float>(), back_inserter(numbers));
		poses.push_back(numbers);
	}
	inf.close();
	return true;
}

bool read_trans(string path, string fname, vector<vector<float>> &trans)
{
	string file = path + fname + ".trans";
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return false;
	}

	trans.clear();
	string line;
	stringstream lineInput;
	while (getline(inf, line))
	{
		lineInput.str("");
		lineInput.clear();
		lineInput << line;
		std::vector<float> numbers(3);
		copy(std::istream_iterator<float>(lineInput), std::istream_iterator<float>(), back_inserter(numbers));
		trans.push_back(numbers);
	}
	inf.close();
	return true;
}


void process_single(int argc, char *argv[])
{
	Eigen::VectorXf beta;
	Eigen::MatrixXf pose;
	Eigen::VectorXf tran;

	vector<float> dat;
	string f_beta = argv[1];
	string f_pose = argv[2];
	read_data(f_beta, dat, 10);
	beta = Eigen::Map<Eigen::VectorXf>(dat.data(), dat.size());
	read_data(f_pose, dat, 72);
	pose = Eigen::Map<Eigen::Matrix<float, 24, 3, Eigen::RowMajor>>(dat.data());

	if (argc == 3)
	{
		tran.resize(3);
		tran.setZero();
	}
	else if (argc == 4)
	{
		string f_tran = argv[3];
		read_data(f_tran, dat, 3);
		tran = Eigen::Map<Eigen::VectorXf>(dat.data(), dat.size());
	}
	else cout << "wrong parameters" << endl;


	SMPL smpl;
	smpl.init();
	smpl.setShape(beta);
	smpl.setPose(pose);

	//Ëæ»ú×ËÌ¬
	//smpl.randomShape();
	//smpl.randomPose();

	smpl.setTrans(tran);
	smpl.updateShape();
	//smpl.test_ceres_pos<float>(1.0);
	smpl.saveMesh("../data/generatedBycode/out.obj");
	cout << "save mesh done" << endl;


}

int main(int argc, char *argv[])
{
	if (argc < 3 || argc > 4)
	{
		cout << "params: (beta.txt pose.txt [tran.txt])" << endl;
		WINPAUSE;
		return 0;
	}


	process_single(argc, argv);
	WINPAUSE;
	return 0;
}
