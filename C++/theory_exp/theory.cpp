#include <iostream>
#include <set>
#include <vector>
#include "Eigen/Dense"
#include <fstream>
#include <algorithm>
#include <random>
#include <iterator>

using namespace std;
using namespace Eigen;

template <typename T>
void printVec(vector<T> &v)
{
    for (auto &x : v)
    {
        cout << x << " ";
    }
}

void simulation(int T, mt19937 &generator, int N, string filename)
{
    // i = -2, j = -1
    vector<int> players(N);
    iota(players.begin(), players.end(), 0);
    VectorXd weights = VectorXd::LinSpaced(N, 0, N - 1); // LinSpaced(size, low, high)
                                                         // our weight = player name exactly!

    VectorXd money = VectorXd::Constant(N, 1);

    MatrixXd result(T, N);

    int k = N / 2;

    ofstream out(filename);
    for (int t = 0; t < T; t++)
    {
        vector<int> subset;

        sample(players.begin(), players.end(), back_inserter(subset), k, generator);
        money(subset) += (money(subset) / money(subset).sum()) * weights(subset).sum();
        // // singleton as well!
        money += weights;
        money(subset) -= weights(subset); // kinda ugly but we don't want to reward this subset twice!

        // cout << ">> subset: ";
        // printVec(subset);
        // cout << endl;
        // cout << "money: \n"
        //      << money << endl;
        result.row(t) = money;
    }

    // cout << "result\n"
    //      << result << endl;
    out << result;
    out.close();
}
int main()
{
    int T = 1000;
    int N = 6;
    int numTrials = 1000;

    #pragma omp parallel for
    for (int trial = 0; trial < numTrials; trial++)
    {
        mt19937 generator(trial);
        simulation(T, generator, N, "theory_" + to_string(trial) + ".txt");
    }
}
