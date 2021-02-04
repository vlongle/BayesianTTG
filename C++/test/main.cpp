#include <iostream>
#include "Eigen/Dense"
using namespace Eigen;

using namespace std;

int main(int argc, char *argv[])
{
    MatrixXd belief(2, 2);
    belief << 1,2,3,4;
    belief(0, 0) = 1.0;
}
