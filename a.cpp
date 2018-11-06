//#include <iostream>
//#include "Eigen/Eigen"
//#include "Eigen/SPQRSupport"
//using namespace Eigen;
//int main() {
//
//	SparseMatrix < double > A(4, 4);
//	std::vector < Triplet < double > > triplets;
//
//	// ��ʼ������Ԫ��
//	int r[3] = { 0, 1, 2 };
//	int c[3] = { 1, 2, 2 };
//	double val[3] = { 6.1, 7.2, 8.3 };
//	for (int i = 0; i < 3; ++i)
//		triplets.push_back(Triplet < double >(r[i], c[i], val[i]));
//
//	// ��ʼ��ϡ�����
//	A.setFromTriplets(triplets.begin(), triplets.end());
//	std::cout << "A = \n" << A << std::endl;
//
//	// һ��QR�ֽ��ʵ��
//	SPQR < SparseMatrix < double > > qr;
//	// ����ֽ�
//	qr.compute(A);
//	// ��һ��A x = b
//	Vector4d b(1, 2, 3, 4);
//	Vector4d x = qr.solve(b);
//	std::cout << "x = \n" << x;
//	std::cout << "A x = \n" << A * x;
//
//	return 0;
//}