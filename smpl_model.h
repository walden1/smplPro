#pragma once

#ifndef _SCL_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#endif // !_SCL_SECURE_NO_WARNINGS


#define REMOVE_DEPENDENCES

#ifdef REMOVE_DEPENDENCES
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iterator> // for istream_iterator

// Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

// 
#define logInfo
#ifdef logInfo
#define logI (printf("--info-- in [%d@%s] ",__LINE__, __FUNCTION__), printf)  
#else
#define logI /\
/logI
#endif

#define logW (printf("--warn-- in [%d@%s] ",__LINE__, __FUNCTION__), printf)  
#define logE (printf("--error- in [%d@%s] ",__LINE__, __FUNCTION__), printf)  

#else
#include "Common.h"
#endif // REMOVE_DEPENDENCES


class SMPL
{
public:
	SMPL();
	~SMPL();

	void init();
	void randomShape();
	void randomPose();
	void resetShape() { beta_.setZero(); }
	void resetPoses() { theta_.setZero(); }
	void setShape(int i, float val) { beta_(i) = val; }
	void setPose(int r, int c, float val) { theta_(r, c) = val; }
	void setShape(const Eigen::VectorXf &beta) { beta_ = beta; }
	void setPose(const Eigen::MatrixXf &theta) { theta_ = theta; }
	void setTrans(const Eigen::Vector3f &trans) { trans_ = trans; }
	void updateShape();
	void updateNormal();

	void restoreJointLocations() { joint_locations_ = joint_weights_ * v_template_; }

	void saveMesh(std::string file);
	void saveMeshWithNormal(std::string file);
	void saveJoints(std::string file);
	void saveParams(std::string file);
	template<typename T> void test_ceres_pos(T temp);

	Eigen::MatrixXf        getVertsTemplate() { return  mapEigenVec2Mat<float>(v_template_, 3, true); }
	Eigen::MatrixXf        getVertsShaped()   { return  mapEigenVec2Mat<float>(v_shaped_, 3, true); }
	const Eigen::VectorXf& getVertsTemplateVec() { return  v_template_; }
	const Eigen::VectorXf& getVertsShapedVec() { return  v_shaped_; }
	const Eigen::VectorXf& getVertsNormalVec() { return  v_normal_; }
	const Eigen::MatrixXi& getFaces() { return faces_; }
	const Eigen::MatrixXf& getJointLocations() { return joint_locations_; }
	const Eigen::MatrixXi& getJointKintree() { return joint_kintree_table_; }
	const Eigen::MatrixXf& getPoseParams() { return theta_; }
	const Eigen::VectorXf& getShapeParam() { return beta_; }
	Eigen::Vector3f        getPoseParams(int i) { return (theta_.row(i)).transpose(); }
	float                  getShapeParam(int i) { return beta_(i); }
	int                    getVertNum() { return v_num_; }
	int                    getFaceNum() { return f_num_; }

private:
	void loadSMPLModel(std::string file);
	void loadTemplateMesh(std::string file);

	Eigen::Matrix3f rodrigues(const Eigen::Vector3f &r);
	Eigen::MatrixXf calcRotMat(const Eigen::MatrixXf &r); // r: (jt_num_,3) matrix, 3 angles for all the vertex
	// return: (3*jt_num_, 3), jt_num_ (3,3) rotation matrix												  
	Eigen::MatrixXf calcJointRot(const Eigen::MatrixXf &R, const Eigen::MatrixXf &J); // R: (3*jt_num_, 3), jt_num_ (3,3) rotation matrix
	// J: (24,3) matrix, positions for all 24 joints
	// return: (24*4,4) matrix, 24 4*4 transformation matrix
	void buildJointKintreeMap();

private:
	void saveShapeAsObjMesh(std::string file, const Eigen::VectorXf &vert, const Eigen::MatrixXi &face);
	void saveShapeAsObjMesh(std::string file, const Eigen::MatrixXf &vert, const Eigen::MatrixXi &face);
	void testEigenMap();
	void testEigenMapFunc();
	template <typename Scalar>
	void readDenseMatrixFromFile(std::string file, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);
	template <typename Scalar>
	void readDenseMatrixFromObj(std::string file, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);
	template <typename Scalar>
	void saveRandomPose2File(std::string file, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);
	template <typename Scalar>
	void saveDenseMatrix2File(std::string file, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat);
	template <typename Scalar>
	void readSparseMatrixFromFile(std::string file, Eigen::SparseMatrix<Scalar> &spmat);
	template<typename Scalar>
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mapEigenVec2Mat(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &vec, int colnum = 3, bool rowmajor = true);
	template<typename Scalar>
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mapEigenMat2Mat(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat, int colnum, bool rowmajor = true);
	template<typename Scalar>
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> mapEigenMat2Vec(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat, bool rowmajor = true);
private:
	Eigen::MatrixXf				shape_bs_;   // shape blend shape (6890*3,10)
	Eigen::MatrixXf				pose_bs_;    // pose blend shape, (6890*3,23*9)
	Eigen::VectorXf				v_template_; // vertex of template mesh, (6890*3,1) [x1,y1,z1,x2,y2,z2,...]
	Eigen::VectorXf				v_shaped_;   // vertex of shaped mesh
	Eigen::VectorXf				v_normal_;   // normal of shaped mesh
	Eigen::MatrixXi				faces_;      // faces of mesh (13776,3)
	Eigen::MatrixXf				skin_weights_;         // skinning weights (6890*24), 24 joints influence on each vertex  
	Eigen::SparseMatrix<float>  joint_weights_;        // joint weights (24,6890), calculate joint location from mesh verteices
	Eigen::MatrixXi			    joint_kintree_table_;  // joint kinetic tree (2,24), root is -1, top row is the father of the bottom row
	Eigen::MatrixXf             joint_locations_;     // joint locations (24,3)
	int							jt_num_;               // joint number (24)
	int							v_num_;                // mesh vertex number (6890)
	int                         f_num_;
	Eigen::VectorXf				beta_;				   // shape parameters (10,1)
	Eigen::MatrixXf				theta_;				   // pose parameters (24,3)
	Eigen::Vector3f             trans_;                // global translation
	std::string                 path_in_ = "../data/m/";// m for male, f for female

};

template<typename Scalar>
inline void SMPL::readDenseMatrixFromFile(std::string file, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return;
	}
#if 0
	int r, c;
	string line, dump;
	stringstream lineInput;
	int cnt = 0;
	while (getline(inf, line))
	{
		lineInput.str("");
		lineInput.clear();
		lineInput << line;
		if (cnt == 0)
		{
			lineInput >> dump >> r >> c;
			mat.resize(r, c);
			continue;    // skip comment lines
		}
		else
		{
			std::vector<Scalar> numbers(c);
			copy(std::istream_iterator<Scalar>(lineInput), std::istream_iterator<Scalar>(), back_inserter(numbers));
			mat.row(cnt - 1) = Map<VectorXi>(numbers.data(), numbers.size());
		}
		cnt++;

	}
	inf.close();

#endif // 0

#if 1
	int r, c;
	std::string line, dump;
	std::stringstream lineInput;
	// parse the first line
	getline(inf, line);
	lineInput.str("");
	lineInput.clear();
	lineInput << line;
	lineInput >> dump >> r >> c;

	// read the rest data and store in a 2d eigen matrix
	std::vector<Scalar> vecdata(r*c);
	std::copy(std::istream_iterator<Scalar>(inf), std::istream_iterator<Scalar>(), vecdata.data());  // ref: https://stackoverflow.com/questions/5005317/c-read-line-of-numbers
	mat = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vecdata.data(), r, c);
	//cout << mat << endl;
	inf.close();
#endif // 1
}

template<typename Scalar>
inline void SMPL::readDenseMatrixFromObj(std::string file, Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return;
	}
#if 1
	int r, c;
	std::string line, dump;
	std::stringstream lineInput;
	int cnt = 0;
	std::string lett2;
	while (getline(inf, line))
	{
		lineInput.str("");
		lineInput.clear();
		lineInput << line;

		
		if (line[0] == '#'){
			if (line.size() > 1){
				lineInput >> dump >> lett2;
				if (lett2 != "" && lett2.substr(0,8) == "Vertices"){
					lineInput >> r;
				}
				else{ continue; }
			}
			else{ continue; }
		}
		else{
			if (cnt == 0)
			{
				c = 3;
				mat.resize(r, c);
			}
			if (line.size()> 0 && line.at(0) == 'v'){
				std::vector<Scalar> numbers(c);
				numbers.clear();
				Scalar x, y, z;
				lineInput >> dump >> x >> y >> z;
				numbers.push_back(x);
				numbers.push_back(y);
				numbers.push_back(z);

				mat.row(cnt) = Eigen::Map<Eigen::VectorXf>(numbers.data(), numbers.size());
				cnt++;
			
			}
			else{ continue; }
		}

		//if (cnt == 0)
		//{
		//	
		//	lineInput >> dump >> r >> c;
		//	mat.resize(r, c);
		//	continue;    // skip comment lines
		//}
		//else
		//{
		//	std::vector<Scalar> numbers(c);
		//	std::copy(std::istream_iterator<Scalar>(lineInput), std::istream_iterator<Scalar>(), back_inserter(numbers));
		//	mat.row(cnt - 1) = Eigen::Map<Eigen::VectorXf>(numbers.data(), numbers.size());
		//}
		//cnt++;

	}
	inf.close();

#endif // 0

#if 0
	int r, c;
	std::string line, dump;
	std::stringstream lineInput;
	// parse the first line
	getline(inf, line);
	lineInput.str("");
	lineInput.clear();
	lineInput << line;
	lineInput >> dump >> r >> c;

	// read the rest data and store in a 2d eigen matrix
	std::vector<Scalar> vecdata(r*c);
	std::copy(std::istream_iterator<Scalar>(inf), std::istream_iterator<Scalar>(), vecdata.data());  // ref: https://stackoverflow.com/questions/5005317/c-read-line-of-numbers
	mat = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vecdata.data(), r, c);
	//cout << mat << endl;
	inf.close();
#endif // 1
}


template<typename Scalar>
inline void SMPL::saveDenseMatrix2File(std::string file, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
	std::ofstream out_file = std::ofstream(file);
	out_file << "dim " << mat.rows() << " " << mat.cols() << std::endl;
	for (int i = 0; i < mat.rows(); i++)
	{
		//out_file << "v " << i + 1 << " ";
		for (int j = 0; j < mat.cols(); j++)
		{
			out_file << mat(i, j) << " ";
		}
		out_file << std::endl;
	}
	out_file.close();
}

template<typename Scalar>
inline void SMPL::saveRandomPose2File(std::string file, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
	std::ofstream out_file = std::ofstream(file);
	//out_file << "pose_theta " << mat.rows() << " " << mat.cols() << std::endl;
	for (int i = 0; i < mat.rows(); i++)
	{
		//out_file << "v " << i + 1 << " ";
		for (int j = 0; j < mat.cols(); j++)
		{
			out_file << mat(i, j) << " ";
		}
		//out_file << std::endl;
	}
	out_file.close();
}


template<typename Scalar>
inline void SMPL::readSparseMatrixFromFile(std::string file, Eigen::SparseMatrix<Scalar>& spmat)
{
	std::ifstream inf;
	inf.open(file.data());
	if (!inf.is_open())
	{
		logE("can't open %s\n", file.c_str());
		return;
	}

	int r, c, i, j;
	Scalar v;
	std::string line, dump;
	std::stringstream lineInput;
	std::vector<T> coefficients;
	int cnt = 0;
	while (getline(inf, line))
	{
		lineInput.str("");
		lineInput.clear();
		lineInput << line;
		if (cnt == 0)
		{
			lineInput >> dump >> r >> c;
		}
		else
		{
			lineInput >> i >> j >> v;
			coefficients.push_back(T(i, j, v));
		}
		cnt++;
	}
	spmat.resize(r, c);
	spmat.setFromTriplets(coefficients.begin(), coefficients.end());
	inf.close();

#if 0
	cout << "nonzeros = " << coefficients.size() << endl;
	for (int k = 0; k<J_regressor.outerSize(); ++k)
		for (Eigen::SparseMatrix<float>::InnerIterator it(J_regressor, k); it; ++it)
		{
			cout << it.row() << " " << it.col() << " " << it.value() << endl;
		}
#endif
}


template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> SMPL::mapEigenVec2Mat(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& vec, int colnum, bool rowmajor)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat;
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> veccopy = vec;
	if (rowmajor)
		mat = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(veccopy.data(), vec.size() / colnum, colnum);
	else
		mat = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(veccopy.data(), vec.size() / colnum, colnum);
	return mat;
}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> SMPL::mapEigenMat2Mat(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat, int colnum, bool rowmajor)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matcopy = mat;
	if (rowmajor)
	{
		matcopy.transposeInPlace();
		return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(matcopy.data(), mat.size() / colnum, colnum);
	}
	else
		//return Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(mat.data(), mat.size() / colnum, colnum);
		return Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(matcopy.data(), mat.size() / colnum, colnum);

}

template<typename Scalar>
inline Eigen::Matrix<Scalar, Eigen::Dynamic, 1> SMPL::mapEigenMat2Vec(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat, bool rowmajor)
{
	Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec;
	Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matcopy = mat;
	if (rowmajor)
	{
		matcopy.transposeInPlace();
		vec = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(matcopy.data(), mat.size());
	}
	else
		vec = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(matcopy.data(), mat.size());
	return vec;
}

