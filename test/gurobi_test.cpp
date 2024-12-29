/* 该例子解决下面的凸二次规划问题：
	minimize x + y + x^2 + x*y + y^2 + y*z + z^2
	subject to x + 2 y + 3 z >= 4
	x + y >= 1
	x, y, z 非负

使用预定义的参数矩阵A和Q进行求解（一般在参数已存入外部数据时使用），否则不推荐使用稠密
矩阵的方法。
*/

#include "gurobi_c++.h"
#include <chrono>
using namespace std;

static bool dense_optimize(GRBEnv* env,
	int rows,
	int cols,
	double* c, /* 目标函数的线性系数项 */
	double* Q, /* 目标函数的二次项 */
	double* A, /* 约束矩阵 */
	char* sense, /* 约束不等关系（大于，小于） */
	double* rhs, /* 右端项向量 */
	double* lb, /* 变量下界 */
	double* ub, /* 变量上界 */
	char* vtype, /* 变量类型 ( continuous , binary , etc .) */
	double* solution,
	double* objvalP)
{
	GRBModel model = GRBModel(*env);
	int i, j;
	bool success = false;
	/* Add variables to the model */
	GRBVar* vars = model.addVars(lb, ub, NULL, vtype, NULL, cols);
	/* Populate A matrix */
	for (i = 0; i < rows; i++) {
		GRBLinExpr lhs = 0;
		for (j = 0; j < cols; j++)
			if (A[i * cols + j] != 0)
				lhs += A[i * cols + j] * vars[j];
		model.addConstr(lhs, sense[i], rhs[i]);
	}
	GRBQuadExpr obj = 0;
	for (j = 0; j < cols; j++)
		obj += c[j] * vars[j];
	for (i = 0; i < cols; i++)
		for (j = 0; j < cols; j++)
			if (Q[i * cols + j] != 0)
				obj += Q[i * cols + j] * vars[i] * vars[j];
	model.setObjective(obj);
	model.optimize();
	model.write(" dense.lp");
	if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
		*objvalP = model.get(GRB_DoubleAttr_ObjVal);
		for (i = 0; i < cols; i++)
			solution[i] = vars[i].get(GRB_DoubleAttr_X);
		success = true;
	}
	delete[] vars;
	return success;
}
int main(int argc,char* argv[])
{
    // CPU 计时
    auto cpu_start = std::chrono::high_resolution_clock::now();
	GRBEnv* env = 0;
	try {
		env = new GRBEnv();
		double c[] = { 1, 1, 0 };
		double Q[3][3] = { {1 , 1, 0}, {0, 1, 1}, {0, 0, 1} };
		double A[2][3] = { {1 , 2, 3}, {1, 1, 0} };
		char sense[] = { '>', '>' };
		double rhs[] = { 4, 1 };
		double lb[] = { 0, 0, 0 };
		bool success;
		double objval, sol[3];
		success = dense_optimize(env, 2, 3, c, &Q[0][0], &A[0][0], sense, rhs,
			lb, NULL, NULL, sol, &objval);
		cout << "x: " << sol[0] << " y: " << sol[1] << " z: " << sol[2] << endl;
	}
	catch (GRBException e) {
		cout << " Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << " Exception during optimization " << endl;
	}
	delete env;
    // CPU 计时结束
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
	return 0;
}
