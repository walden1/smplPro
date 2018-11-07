#include "smpl_model.h"

SMPL::SMPL()
{
	//testEigenMap();
	//testEigenMapFunc();

}

SMPL::~SMPL()
{
}

void SMPL::init()
{
	logI("-------------------------------init...\n");
	loadSMPLModel("");
	loadTemplateMesh("");
	buildJointKintreeMap();

	v_shaped_ = v_template_;
	v_normal_ = Eigen::VectorXf::Zero(v_num_ * 3);

	beta_.resize(10);     beta_.setZero();
	theta_.resize(24, 3); theta_.setZero();
	trans_.setZero();
	joint_locations_.resize(24, 3);
	logI("-------------------------------init done.\n");
}

void SMPL::loadSMPLModel(std::string file)
{
	readDenseMatrixFromFile<float>(path_in_ + "pose_bs.txt", pose_bs_);	          logI("read pose blendshape done.\n");
	readDenseMatrixFromFile<float>(path_in_ + "shape_bs.txt", shape_bs_);         logI("read shape blendshape done.\n");
	readDenseMatrixFromFile<float>(path_in_ + "skin_weights.txt", skin_weights_); logI("read skin_weights done.\n");
	readDenseMatrixFromFile<int>(path_in_ + "joint_kintree.txt", joint_kintree_table_); logI("read joint_kintree done.\n");
	readSparseMatrixFromFile<float>(path_in_ + "joint_weights.txt", joint_weights_);    logI("read joint_weights done.\n");
	jt_num_ = joint_kintree_table_.cols();
}

void SMPL::loadTemplateMesh(std::string file)
{
	Eigen::MatrixXf vert;
	//readDenseMatrixFromFile<float>(path_in_ + "template_mesh_v.txt", vert);	logI("read template_mesh_v done.\n");
	//readDenseMatrixFromObj<float>(path_in_ + "template_mesh_修改.obj", vert);	logI("read template_mesh_v done.\n");
	readDenseMatrixFromObj<float>(path_in_ + "T_template_mesh_edit_scale1.obj", vert);	logI("read template_mesh_v done.\n");
	v_num_ = vert.rows();
	v_template_ = mapEigenMat2Vec<float>(vert, true); // row major map, x1, y1, z1, x2, y2, z2, ...
	readDenseMatrixFromFile<int>(path_in_ + "template_mesh_f.txt", faces_);	logI("read template_mesh_f done.\n");
	f_num_ = faces_.rows();
}

void SMPL::randomShape()
{
	beta_.setRandom();			// [-1,1]
	beta_ = 0.5*(beta_ + Eigen::VectorXf::Ones(beta_.size()));  // [0,1]
	beta_ = (beta_ - 0.5*Eigen::VectorXf::Ones(beta_.size()))*0.06; //[-0.03, 0.03]
}


void SMPL::randomPose()
{
	theta_.setRandom();
	theta_ = 0.5*(theta_ + Eigen::MatrixXf::Ones(theta_.rows(), theta_.cols()));
	theta_ = (theta_ - 0.5*Eigen::MatrixXf::Ones(theta_.rows(), theta_.cols()))*0.4; //[-0.02, 0.02]

	//static int count = 0;
	//char file_name[256];
	//sprintf_s(file_name, "../data/random_pose_%d.txt", count);

	//std::stringstream ss;
	//std::string file_name_str;
	//ss.clear();
	//ss.str("");
	//ss << file_name;
	//ss >> file_name_str;

	////保存随机pose
	//saveRandomPose2File<float>(file_name_str, theta_);
	//count++;
}

void SMPL::updateShape()
{
	//saveDenseMatrix2File<float>("../param_beta.txt", beta_);
	//saveDenseMatrix2File<float>("../param_theta.txt", theta_);

	Eigen::VectorXf v_shaped = shape_bs_ * beta_ + v_template_;				// Ts = T + Bs = T + s * belta   
//	v_shaped = v_template_;

	Eigen::MatrixXf J = joint_weights_ * mapEigenVec2Mat<float>(v_shaped, 3, true);		//计算joint坐标

	//saveShapeAsObjMesh("../joint_pos.obj", J, Eigen::MatrixXi());

	Eigen::MatrixXf R = calcRotMat(theta_);			//将theta 转为 旋转矩阵 24*3 --> 216个元素的旋转矩阵 24*9 ---> 23 * 9 = 207
	int n = theta_.rows();
	Eigen::MatrixXf I(3 * n - 3, 3);
	for (int i = 0; i < n - 1; i++)
		I.block(3 * i, 0, 3, 3) = Eigen::MatrixXf::Identity(3, 3);
	I = (R.bottomRows(3 * n - 3) - I);

	Eigen::VectorXf lrotmin = mapEigenMat2Vec<float>(I, true);
	Eigen::VectorXf v_posed = pose_bs_ * lrotmin + v_shaped;				// T + Bs + Bp = T + s * belta + (Rn-R0) * p ( = Ts + Bp )
//	v_posed = v_template_;
	v_posed = v_shaped;

	//蒙皮
	Eigen::MatrixXf G = calcJointRot(R, J);
	Eigen::MatrixXf T = skin_weights_ * mapEigenMat2Mat(G, 16, true);  // (6890,16) transformation matrix for all vertex 计算每个顶点的变换矩阵

	for (int i = 0; i < v_num_; i++)
	{
		Eigen::VectorXf Tvec = T.row(i);
		Eigen::MatrixXf Tmat = mapEigenVec2Mat(Tvec, 4, true);
		Eigen::Vector4f vpose;
		vpose << v_posed(3 * i), v_posed(3 * i + 1), v_posed(3 * i + 2), 1.0;
		vpose = Tmat * vpose;
		v_shaped_(3 * i) = vpose(0) + trans_(0);						//v_shape 蒙皮变形后的模型
		v_shaped_(3 * i + 1) = vpose(1) + trans_(1);
		v_shaped_(3 * i + 2) = vpose(2) + trans_(2);
	}
	for (int i = 0; i < jt_num_; i++) // update joint locations
	{
		Eigen::Vector4f jp0 = Eigen::Vector4f(J(i, 0), J(i, 1), J(i, 2), 1.0);
		Eigen::Vector4f jp = G.block(4 * i, 0, 4, 4)*jp0;
		//joint_locations_.row(i) = (jp.block(0, 0, 3, 1)).transpose() - trans_;
		joint_locations_.row(i) = ( jp.block(0, 0, 3, 1) - trans_ ).transpose();
	}
	saveShapeAsObjMesh("../data/generatedBycode/joint_pos1.obj", joint_locations_, Eigen::MatrixXi());
	saveShapeAsObjMesh("../data/generatedBycode/v_template.obj", v_template_, faces_);
	saveShapeAsObjMesh("../data/generatedBycode/v_shaped.obj", v_shaped, faces_);
	saveShapeAsObjMesh("../data/generatedBycode/v_posed.obj", v_posed, faces_);
	saveShapeAsObjMesh("../data/generatedBycode/v_shape_.obj", v_shaped_, faces_);	//最终lbs变形结果
	saveDenseMatrix2File<float>("../data/generatedBycode/v_template.txt", v_template_);
	saveDenseMatrix2File<float>("../data/generatedBycode/v_shaped.txt", v_shaped);
	saveDenseMatrix2File<float>("../data/generatedBycode/v_posed.txt", v_posed);
	saveDenseMatrix2File<float>("../data/generatedBycode/v_shape_.txt", v_shaped_);
}

template<typename T>
inline void Mat4Multiply(T *a, T *b, T *res)  // column-major
{
	res[0] = a[0] * b[0] + a[4] * b[1] + a[8] * b[2] + a[12] * b[3];
	res[1] = a[1] * b[0] + a[5] * b[1] + a[9] * b[2] + a[13] * b[3];
	res[2] = a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3];
	res[3] = a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3];

	res[4] = a[0] * b[4] + a[4] * b[5] + a[8] * b[6] + a[12] * b[7];
	res[5] = a[1] * b[4] + a[5] * b[5] + a[9] * b[6] + a[13] * b[7];
	res[6] = a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7];
	res[7] = a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7];

	res[8] = a[0] * b[8] + a[4] * b[9] + a[8] * b[10] + a[12] * b[11];
	res[9] = a[1] * b[8] + a[5] * b[9] + a[9] * b[10] + a[13] * b[11];
	res[10] = a[2] * b[8] + a[6] * b[9] + a[10] * b[10] + a[14] * b[11];
	res[11] = a[3] * b[8] + a[7] * b[9] + a[11] * b[10] + a[15] * b[11];

	res[12] = a[0] * b[12] + a[4] * b[13] + a[8] * b[14] + a[12] * b[15];
	res[13] = a[1] * b[12] + a[5] * b[13] + a[9] * b[14] + a[13] * b[15];
	res[14] = a[2] * b[12] + a[6] * b[13] + a[10] * b[14] + a[14] * b[15];
	res[15] = a[3] * b[12] + a[7] * b[13] + a[11] * b[14] + a[15] * b[15];
}

template<typename T>
void vec10Multiply(Eigen::RowVectorXf mat1, T * mat2, T &res){
	T sum(0);
	for (int i = 0; i < 10; i++){
		sum = sum + T(mat1[i]) * mat2[i];
	}
	res = sum;
}

template<typename T>
void SMPL::test_ceres_pos(T temp){

	Eigen::MatrixXf * shape_bs_ptr_ = &shape_bs_;
	Eigen::MatrixXi	* joint_kintree_table_ptr_ = &joint_kintree_table_;
	//---------------
	Eigen::VectorXf v_shaped = shape_bs_ * beta_ + v_template_;
	
	Eigen::MatrixXf J = joint_weights_ * mapEigenVec2Mat<float>(v_shaped, 3, true);
	Eigen::MatrixXf * skin_weights_ptr_ = &skin_weights_;

	T x_belta[10] = { T(0.0) };	for (int i = 0; i < 10; i++){ x_belta[i] = beta_(i); }
	T x_theta[72] = { T(0.0) };	
	for (int i = 0; i < 24; i++)
	{ 
		x_theta[3 * i + 0] = theta_( i, 0); 
		x_theta[3 * i + 1] = theta_( i, 1);
		x_theta[3 * i + 2] = theta_( i, 2);
	}
	T x_trans[3] = { T(0.0) };	for (int i = 0; i < 3; i++){ x_trans[i] = trans_(i); }

	for (int index_ = 0; index_ < v_num_; index_++)
	{
		Eigen::Vector3f v_template_pose_(v_template_(3 * index_ + 0), v_template_(3 * index_ + 1), v_template_(3 * index_ + 2));

		T vertex_pos_bef[3] = { T(0.0) };	
		
		vec10Multiply<T>(shape_bs_ptr_->row(3 * index_ + 0), x_belta, vertex_pos_bef[0]);
		vec10Multiply<T>(shape_bs_ptr_->row(3 * index_ + 1), x_belta, vertex_pos_bef[1]);
		vec10Multiply<T>(shape_bs_ptr_->row(3 * index_ + 2), x_belta, vertex_pos_bef[2]);
		vertex_pos_bef[0] = vertex_pos_bef[0] + T(v_template_pose_(0));
		vertex_pos_bef[1] = vertex_pos_bef[1] + T(v_template_pose_(1));
		vertex_pos_bef[2] = vertex_pos_bef[2] + T(v_template_pose_(2));

		T R[24][9] = { T(0.0) };
		for (int i = 0; i < 24; i++){
			T * rotMat_i = new T[9];
			T angle_axis[3] = { x_theta[3 * i + 0], x_theta[3 * i + 1], x_theta[3 * i + 2] };
			Eigen::Vector3f a_axis(angle_axis[0], angle_axis[1], angle_axis[2]);
			Eigen::Matrix3f R1 = rodrigues(a_axis);
			rotMat_i[0] = R1(0, 0); rotMat_i[1] = R1(1, 0); rotMat_i[2] = R1(2, 0);
			rotMat_i[3] = R1(0, 1); rotMat_i[4] = R1(1, 1); rotMat_i[5] = R1(2, 1);
			rotMat_i[6] = R1(0, 2); rotMat_i[7] = R1(1, 2); rotMat_i[8] = R1(2, 2);
			//ceres::AngleAxisToRotationMatrix(angle_axis, rotMat_i);
			for (int j = 0; j < 9; j++){
				R[i][j] = rotMat_i[j];
			}
		}

		T G[24][16] = { T(0.0) };
		G[0][0] = R[0][0]; G[0][1] = R[0][1]; G[0][2] = R[0][2];
		G[0][4] = R[0][3]; G[0][5] = R[0][4]; G[0][6] = R[0][5];
		G[0][8] = R[0][6]; G[0][9] = R[0][7]; G[0][10] = R[0][8];
		G[0][12] = T(J(0, 0)); G[0][13] = T(J(0, 1)); G[0][14] = T(J(0, 2));
		G[0][15] = T(1.0);

		for (int jt = 1; jt < 24; jt++){
			int parent = joint_kintree_table_ptr_->coeffRef(0, jt);
			T Gi[16] = { T(0.0) };
			Gi[0] = R[jt][0]; Gi[1] = R[jt][1]; Gi[2] = R[jt][2];
			Gi[4] = R[jt][3]; Gi[5] = R[jt][4]; Gi[6] = R[jt][5];
			Gi[8] = R[jt][6]; Gi[9] = R[jt][7]; Gi[10] = R[jt][8];
			Gi[12] = T(J(jt, 0) - J(parent, 0));
			Gi[13] = T(J(jt, 1) - J(parent, 1));
			Gi[14] = T(J(jt, 2) - J(parent, 2));
			Gi[15] = T(1.0);

			T res[16] = { T(0.0) };
			Mat4Multiply<T>(G[parent], Gi, res);

			for (int i = 0; i < 16; i++)  { G[jt][i] = res[i]; }

		}

		for (int jt = 0; jt < 24; jt++){
			T J0[4] = { T(0.0f) };
			J0[0] = T(J(jt, 0)); J0[1] = T(J(jt, 1)); J0[2] = T(J(jt, 2));

			T z0 = G[jt][0] * J0[0] + G[jt][4] * J0[1] + G[jt][8] * J0[2];
			T z1 = G[jt][1] * J0[0] + G[jt][5] * J0[1] + G[jt][9] * J0[2];
			T z2 = G[jt][2] * J0[0] + G[jt][6] * J0[1] + G[jt][10] * J0[2];

			G[jt][12] = G[jt][12] - z0;
			G[jt][13] = G[jt][13] - z1;
			G[jt][14] = G[jt][14] - z2;
		}


		Eigen::RowVectorXf s_weight = skin_weights_ptr_->row(index_);
		T vertex_transform[16] = { T(0.0) };

		//方案1
		for (int i = 0; i < 16; i++){
			for (int j = 0; j < 24; j++){
				vertex_transform[i] = vertex_transform[i] + T(s_weight[j]) * G[j][i];
			}
		}

		T vertex_pos_aft[3] = { T(0.0) };
		vertex_pos_aft[0] =
			vertex_transform[0] * vertex_pos_bef[0] + vertex_transform[4] * vertex_pos_bef[1] +
			vertex_transform[8] * vertex_pos_bef[2] + vertex_transform[12] + x_trans[0];
		vertex_pos_aft[1] =
			vertex_transform[1] * vertex_pos_bef[0] + vertex_transform[5] * vertex_pos_bef[1] +
			vertex_transform[9] * vertex_pos_bef[2] + vertex_transform[13] + x_trans[1];
		vertex_pos_aft[2] =
			vertex_transform[2] * vertex_pos_bef[0] + vertex_transform[6] * vertex_pos_bef[1] +
			vertex_transform[10] * vertex_pos_bef[2] + vertex_transform[14] + x_trans[2];

		//方案2
		//^T
		/*T G_T[24][16] = { T(0.0) };
		for (int jt = 0; jt < 24; jt++){
			G_T[jt][0] = G[jt][0];  G_T[jt][1] = G[jt][4]; G_T[jt][2] = G[jt][8];  G_T[jt][3] = G[jt][12];
			G_T[jt][4] = G[jt][1];  G_T[jt][5] = G[jt][5]; G_T[jt][6] = G[jt][9];  G_T[jt][7] = G[jt][13];
			G_T[jt][8] = G[jt][2];  G_T[jt][9] = G[jt][6]; G_T[jt][10] = G[jt][10]; G_T[jt][11] = G[jt][14];
			G_T[jt][12] = G[jt][3];  G_T[jt][13] = G[jt][7]; G_T[jt][14] = G[jt][11]; G_T[jt][15] = G[jt][15];
		}

		for (int i = 0; i < 16; i++){
			for (int j = 0; j < 24; j++){
				vertex_transform[i] = vertex_transform[i] + T(s_weight[j]) * G_T[j][i];
			}
		}

		T vertex_pos_aft[3] = { T(0.0) };
		vertex_pos_aft[0] =
			vertex_transform[0] * vertex_pos_bef[0] + vertex_transform[1] * vertex_pos_bef[1] +
			vertex_transform[2] * vertex_pos_bef[2] + vertex_transform[3] + x_trans[0];
		vertex_pos_aft[1] =
			vertex_transform[4] * vertex_pos_bef[0] + vertex_transform[5] * vertex_pos_bef[1] +
			vertex_transform[6] * vertex_pos_bef[2] + vertex_transform[7] + x_trans[1];
		vertex_pos_aft[2] =
			vertex_transform[8] * vertex_pos_bef[0] + vertex_transform[9] * vertex_pos_bef[1] +
			vertex_transform[10] * vertex_pos_bef[2] + vertex_transform[11] + x_trans[2];*/
		
		//end
		

		//v_shape 蒙皮变形后的模型
		v_shaped_(3 * index_ + 0) = vertex_pos_aft[0];						
		v_shaped_(3 * index_ + 1) = vertex_pos_aft[1];
		v_shaped_(3 * index_ + 2) = vertex_pos_aft[2];

	}

}
template void SMPL::test_ceres_pos<float>(float);

void SMPL::updateNormal()
{
	v_normal_.setZero();
	for (int i = 0; i < f_num_; i++)
	{
		Eigen::Vector3i face = faces_.row(i);
		const Eigen::Vector3f &p0 = Eigen::Vector3f(v_shaped_(3 * face(0)), v_shaped_(3 * face(0) + 1), v_shaped_(3 * face(0) + 2));
		const Eigen::Vector3f &p1 = Eigen::Vector3f(v_shaped_(3 * face(1)), v_shaped_(3 * face(1) + 1), v_shaped_(3 * face(1) + 2));
		const Eigen::Vector3f &p2 = Eigen::Vector3f(v_shaped_(3 * face(2)), v_shaped_(3 * face(2) + 1), v_shaped_(3 * face(2) + 2));
		Eigen::Vector3f a = p0 - p1;
		Eigen::Vector3f b = p1 - p2;
		Eigen::Vector3f c = p2 - p0;
		float l2a = a.squaredNorm();
		float l2b = b.squaredNorm();
		float l2c = c.squaredNorm();
		Eigen::Vector3f facenormal = a.cross(b);
		for (int j = 0; j < 3; j++)
		{
			v_normal_(face(0) + j) += facenormal(j) * (1.0f / (l2a * l2c));
			v_normal_(face(1) + j) += facenormal(j) * (1.0f / (l2b * l2a));
			v_normal_(face(2) + j) += facenormal(j) * (1.0f / (l2c * l2b));
		}
	}

	// Make them all unit-length
	for (int i = 0; i < v_num_; i++)
	{
		Eigen::Vector3f N = Eigen::Vector3f(v_normal_(3 * i), v_normal_(3 * i + 1), v_normal_(3 * i + 2));
		N.normalize();
		v_normal_(3 * i) = N(0);
		v_normal_(3 * i + 1) = N(1);
		v_normal_(3 * i + 2) = N(2);
		continue;

		float norm = N.norm();
		if (norm > 1e-6)
		{
			v_normal_(3 * i) = N(0) / norm;
			v_normal_(3 * i + 1) = N(1) / norm;
			v_normal_(3 * i + 2) = N(2) / norm;
		}
		else
		{
			v_normal_(3 * i) = 1.0;
			v_normal_(3 * i + 1) = 0.0;
			v_normal_(3 * i + 2) = 0.0;
		}
	}
	//logI("update normal done\n");
}

void SMPL::saveMesh(std::string file)
{
	saveShapeAsObjMesh(file, v_shaped_, faces_);
	logI("save shaped mesh to %s done\n", file.c_str());
}

void SMPL::saveMeshWithNormal(std::string file)
{
	std::ofstream out_file = std::ofstream(file);
	out_file << "# v " << v_num_ << " f " << f_num_ << std::endl;
	for (int i = 0; i < v_shaped_.size(); i = i + 3)
	{
		out_file << "v ";
		out_file << v_shaped_(i) << " " << v_shaped_(i + 1) << " " << v_shaped_(i + 2) << std::endl;
		out_file << "vn ";
		out_file << v_normal_(i) << " " << v_normal_(i + 1) << " " << v_normal_(i + 2) << std::endl;
	}

	for (int i = 0; i < faces_.rows(); i++)
	{
		out_file << "f";
		for (int j = 0; j < faces_.cols(); j++)
		{
			out_file << " " << faces_(i, j) + 1;
		}
		out_file << std::endl;
	}
	out_file.close();
	logI("save shaped mesh with normals to %s done\n", file.c_str());
}

void SMPL::saveJoints(std::string file)
{
	saveShapeAsObjMesh(file, joint_locations_, Eigen::MatrixXi());
	logI("save joint locations to %s done\n", file.c_str());
}

void SMPL::saveParams(std::string file)
{
	std::ofstream out_file = std::ofstream(file);
	out_file << "shape " << std::endl;
	for (int i = 0; i < beta_.size(); i++)
	{
		out_file << beta_(i) << " ";
	}
	out_file << std::endl;

	out_file << "pose " << theta_.rows() << " " << theta_.cols() << std::endl;
	for (int i = 0; i < theta_.rows(); i++)
	{
		for (int j = 0; j < theta_.cols(); j++)
		{
			out_file << theta_(i, j) << " ";
		}
		out_file << std::endl;
	}
	out_file << std::endl;
	out_file.close();

	logI("save pose & shape parameters to %s done\n", file.c_str());
}

Eigen::Matrix3f SMPL::rodrigues(const Eigen::Vector3f & r)
{
	// ref: https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#void%20Rodrigues(InputArray%20src,%20OutputArray%20dst,%20OutputArray%20jacobian)
	float theta = r.norm();
	if (abs(theta) < 0.0001)
		return Eigen::Matrix3f::Identity(3, 3);

	Eigen::Vector3f s = r.normalized();
	Eigen::Matrix3f T;
	T << 0, -s(2), s(1),
		s(2), 0, -s(0),
		-s(1), s(0), 0;
	Eigen::Matrix3f R = cosf(theta)*Eigen::Matrix3f::Identity(3, 3) + (1 - cosf(theta))*s*(s.transpose()) + sinf(theta) * T;

	//cout << R << endl;
	return R;
}

Eigen::MatrixXf SMPL::calcRotMat(const Eigen::MatrixXf & r)
{
	int n = r.rows();
	Eigen::MatrixXf R(3 * n, 3);
	for (int i = 0; i < n; i++)
		R.block(3 * i, 0, 3, 3) = rodrigues(r.row(i));
	return R;
}

Eigen::MatrixXf SMPL::calcJointRot(const Eigen::MatrixXf &R, const Eigen::MatrixXf &J)
{
	Eigen::MatrixXf G = Eigen::MatrixXf::Zero(4 * jt_num_, 4);
	G.block(0, 0, 3, 3) = R.block(0, 0, 3, 3);  // rotation
	G.block(0, 3, 3, 1) = (J.row(0)).transpose();  // translation
	G(3, 3) = 1.0f;

	for (int jt = 1; jt < jt_num_; jt++)
	{
		int parent = joint_kintree_table_(0, jt);
		Eigen::Matrix4f Gi = Eigen::Matrix4f::Zero();
		Gi.block(0, 0, 3, 3) = R.block(3 * jt, 0, 3, 3);  // rotation
		Gi.block(0, 3, 3, 1) = (J.row(jt) - J.row(parent)).transpose();  // translation
		Gi(3, 3) = 1.0f;
		G.block(4 * jt, 0, 4, 4) = G.block(4 * parent, 0, 4, 4)*Gi;
	}

	for (int jt = 0; jt < jt_num_; jt++)
	{
		Eigen::MatrixXf Gi = G.block(4 * jt, 0, 4, 4);  // 4*4

		Eigen::Vector4f J0;
		J0 << J(jt, 0), J(jt, 1), J(jt, 2), 0.0f;

		Eigen::Matrix4f G0 = Eigen::Matrix4f::Identity();
		G0.block(0, 3, 4, 1) = J0; G0(3, 3) = 1.0f;
		G.block(4 * jt, 0, 4, 4) = Gi*G0.inverse();

		/*std::cout << "Gi\n" << Gi << std::endl;
		std::cout << "G0.inverse:\n" << G0.inverse() << "\n"<<std::endl;
		std::cout << "***   G.block   =  Gi * G0.inverse()\n"<<G.block(4 * jt, 0, 4, 4) << std::endl;*/

		//Eigen::Matrix4f Z0 = Eigen::Matrix4f::Zero();
		//Z0.col(3) = Gi * J0;
		//G.block(4 * jt, 0, 4, 4) = Gi - Z0;

		/*std::cout << "-------------------------------------------------------" << std::endl;*/

	}
	return G;
}

void SMPL::buildJointKintreeMap()
{
	std::map<int, std::vector<int>>  joint_kintree_map;    // joint id -> joint parents with order

	int jt_num = joint_kintree_table_.cols();
	for (int jt_i = 0; jt_i < jt_num; jt_i++)
	{
		std::vector<int> parents;
		int parent = -1;
		int jt_j = jt_i;
		while (joint_kintree_table_(0, jt_j) != -1)
		{
			parent = joint_kintree_table_(0, jt_j);
			parents.push_back(parent);
			jt_j = parent;
		}
		joint_kintree_map.insert(std::map<int, std::vector<int>>::value_type(jt_i, parents));
	}

#if 0  // print out to screen
	std::map<int, std::vector<int>>::iterator iter;
	for (iter = joint_kintree_map.begin(); iter != joint_kintree_map.end(); iter++)
	{
		std::cout << iter->first << ": ";
		std::vector<int> par = iter->second;
		for (int i = 0; i < par.size(); i++)
			std::cout << " " << par[i];
		std::cout << std::endl;
	}

#endif
}

void SMPL::saveShapeAsObjMesh(std::string file, const Eigen::VectorXf &vert, const Eigen::MatrixXi &face)
{
	std::ofstream out_file = std::ofstream(file);
	out_file << "# v " << vert.size() / 3 << " f " << face.rows() << std::endl;
	for (int i = 0; i < vert.rows(); i = i + 3)
	{
		out_file << "v ";
		out_file << vert(i) << " " << vert(i + 1) << " " << vert(i + 2) << std::endl;
	}

	for (int i = 0; i < face.rows(); i++)
	{
		out_file << "f";
		for (int j = 0; j < face.cols(); j++)
		{
			out_file << " " << face(i, j) + 1;
		}
		out_file << std::endl;
	}
	out_file.close();
}

void SMPL::saveShapeAsObjMesh(std::string file, const Eigen::MatrixXf &vert, const Eigen::MatrixXi &face)
{
	std::ofstream out_file = std::ofstream(file);
	out_file << "# v " << vert.size() / 3 << " f " << face.rows() << std::endl;
	for (int i = 0; i < vert.rows(); i++)
	{
		out_file << "v ";
		out_file << vert(i, 0) << " " << vert(i, 1) << " " << vert(i, 2) << std::endl;
	}

	for (int i = 0; i < face.rows(); i++)
	{
		out_file << "f";
		for (int j = 0; j < face.cols(); j++)
		{
			out_file << " " << face(i, j) + 1;
		}
		out_file << std::endl;
	}
	out_file.close();
}

void SMPL::testEigenMap()
{
	Eigen::MatrixXf mat(2, 3);
	mat << 0, 1, 2,
		3, 4, 5;
	std::cout << mat << std::endl << std::endl;

	Eigen::VectorXf vec1 = Eigen::Map<Eigen::VectorXf, Eigen::RowMajor>(mat.data(), mat.size());
	std::cout << "vec1\n" << vec1 << std::endl << std::endl;

	Eigen::VectorXf vec2 = Eigen::Map<Eigen::VectorXf>((mat.transpose()).data(), mat.size());
	std::cout << "vec2\n" << vec2 << std::endl << std::endl;

	mat.transposeInPlace();
	Eigen::VectorXf vec3 = Eigen::Map<Eigen::VectorXf>(mat.data(), mat.size());
	std::cout << "vec3\n" << vec3 << std::endl << std::endl;

	std::cout << mat << std::endl << std::endl;

	Eigen::MatrixXf mat2 = Eigen::Map<Eigen::MatrixXf>(vec1.data(), 2, 3);
	std::cout << mat2 << std::endl << std::endl;

	Eigen::MatrixXf mat3 = Eigen::Map<Eigen::MatrixXf>(vec1.data(), 2, 3);
	std::cout << mat3 << std::endl << std::endl;
}

void SMPL::testEigenMapFunc()
{
	Eigen::MatrixXf mat(2, 3);
	mat << 0, 1, 2,
		3, 4, 5;
	std::cout << "A \n" << mapEigenMat2Vec<float>(mat, true) << std::endl << std::endl;
	std::cout << "B \n" << mapEigenMat2Vec<float>(mat, false) << std::endl << std::endl;
	std::cout << "C \n" << mapEigenMat2Mat<float>(mat, 2, true) << std::endl << std::endl;
	std::cout << "D \n" << mapEigenMat2Mat<float>(mat, 2, false) << std::endl << std::endl;
	std::cout << "---" << std::endl;

	Eigen::VectorXi vec(6);
	vec << 0, 1, 2, 3, 4, 5, 6;
	std::cout << mapEigenVec2Mat<int>(vec, 2, true) << std::endl << std::endl;
	std::cout << mapEigenVec2Mat<int>(vec, 2, false) << std::endl << std::endl;
}