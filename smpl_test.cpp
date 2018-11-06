#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <Eigen\src\IterativeLinearSolvers\ConjugateGradient.h>



using namespace std;
using namespace Eigen;

//�����в���
void build_fvec(Eigen::VectorXd &x, Eigen::VectorXd &fvec){

	fvec[0] = 1.0 / sqrt(3.0) * (x[0] - 10);
	fvec[1] = 1.0 / sqrt(5.0) * (x[1] - 7);
	fvec[2] = 1.0 / 3.0 * (x[2] - 0.5);

}

//����Jacobian����-�ؼ�
void build_jacobian(Eigen::VectorXd &x, Eigen::MatrixXd &fjac){
	fjac.coeffRef(0, 0) = 1.0 / sqrt(3.0);
	fjac.coeffRef(1, 1) = 1.0 / sqrt(5.0);
	fjac.coeffRef(2, 2) = 1.0 / 3.0;
}

//��� f(x0,x1,x2) = 1/3 (x0-10)^2 + 1/5 (x1-7)^2 + 1/9 (x2-0.5)^2 ������Сʱ������ֵ
double gauss_newton_solver(){

	Eigen::VectorXd pre_fvec, pre_x, best_x;
	double pre_energy = 0, cur_energy = 0, best_energy = 1e10;

	int residual_rows = 3;
	int jacobian_cols = 3;

	Eigen::VectorXd x(3);
	x.setZero();

	//�в�
	Eigen::VectorXd fvec(residual_rows);
	fvec.setZero();
	build_fvec(x, fvec);

	cur_energy = best_energy = sqrt(fvec.dot(fvec))*0.5;
	cout << "��ʼ������" << cur_energy << endl;

	//Jacobian��ʼ��
	Eigen::MatrixXd fjac(residual_rows, jacobian_cols);
	fjac.setZero();


	//���
	Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower | Eigen::Upper, Eigen
		::IdentityPreconditioner> solver_cg;
	solver_cg.setMaxIterations(100);

	int iter = 0;
	while (iter<10){
		cout << "\niter:" << iter << endl;
		//Jacobian���ݹ���
		build_jacobian(x, fjac);

		Eigen::MatrixXd A = fjac.transpose();
		Eigen::VectorXd b = A * fvec;
		A = A*fjac;

		Eigen::VectorXd delta = solver_cg.compute(A).solve(b);
		cout << "\t error:" << solver_cg.error() << endl;

		x = x - delta;
		pre_fvec = fvec;
		pre_energy = cur_energy;

		//���²в��ǰ����ֵ
		build_fvec(x, fvec);
		cur_energy = sqrt(fvec.dot(fvec))*0.5;

		if (cur_energy < best_energy){
			best_energy = cur_energy;
			best_x = x;
		}

		cout << "\tĿǰ�������:" << best_energy << "   ��ǰ����:" << cur_energy << "\n" << endl;

		++iter;

		if (iter >= 2){
			if (fabs(cur_energy - pre_energy) < (1e-6)*(1 + cur_energy)) break;

			Eigen::VectorXd gradient = fjac.transpose() * pre_fvec;
			double gradient_max = gradient.maxCoeff();
			if (gradient_max < (1e-2)*(1 + cur_energy)) break;
		}
		double delta_max = delta.maxCoeff();
		if (delta_max<(1e-3)*(1 + delta_max)) break;


	}

	cout << "�������:\n" << best_energy << endl;
	x = best_x;
	cout << "������:\n" << x << endl;
	return best_energy;

}

//
//int main(){
//	gauss_newton_solver();
//	return 0;
//
//}