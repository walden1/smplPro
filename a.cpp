//#include <iostream>
//#include "Eigen/Eigen"
//#include "Eigen/SPQRSupport"
//using namespace Eigen;
//int main() {
//
//	SparseMatrix < double > A(4, 4);
//	std::vector < Triplet < double > > triplets;
//
//	// 初始化非零元素
//	int r[3] = { 0, 1, 2 };
//	int c[3] = { 1, 2, 2 };
//	double val[3] = { 6.1, 7.2, 8.3 };
//	for (int i = 0; i < 3; ++i)
//		triplets.push_back(Triplet < double >(r[i], c[i], val[i]));
//
//	// 初始化稀疏矩阵
//	A.setFromTriplets(triplets.begin(), triplets.end());
//	std::cout << "A = \n" << A << std::endl;
//
//	// 一个QR分解的实例
//	SPQR < SparseMatrix < double > > qr;
//	// 计算分解
//	qr.compute(A);
//	// 求一个A x = b
//	Vector4d b(1, 2, 3, 4);
//	Vector4d x = qr.solve(b);
//	std::cout << "x = \n" << x;
//	std::cout << "A x = \n" << A * x;
//
//	return 0;
//}