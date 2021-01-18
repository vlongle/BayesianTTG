#include "Utils.hpp"

// softmax for matrix
//MatrixXd softmax(const MatrixXd &M)
//{
//    int numberOfRows = M.rows();
//    int numberOfCols = M.cols();
//    MatrixXd ret(numberOfRows, numberOfCols);
//    for (int i = 0; i < numberOfRows; i++)
//    {
//        // https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
//        // trick for stability. We'll subtract/add the largest number (in absolute value) to each entry before
//        // applying softmax (not used here but something to keep in mind)
//        ret.row(i) = M.row(i).array().exp();
//        ret.row(i) /= ret.row(i).sum();
//    }
//    return ret;
//}
//

// softmax for vector

VectorXd softmax(const VectorXd &v)
{
    VectorXd ret = v.array().exp();
    ret /= ret.sum();
    return ret;
}

MatrixXd generateDivisionRule(int numPlayers)
{
    int m = numPlayers - 1; // no. dimension - 1
    int n = 10;             // no. subintervals in the simplex
    // e.g. if m=2, n=3 then points are
    // [0,0,3], [0,1,2], ..., [3,0,0]

    // increment step of 10% demand
    int ng = simplex_grid_size(m, n);
    int *ret = simplex_grid_index_all(m, n, ng);
    MatrixXd divisionRule(ng, m + 1);
    for (int i = 0; i < ng * (m + 1); i++)
    {
        divisionRule(i / (m + 1), i % (m + 1)) = ret[i] / 10.0;
    }
    return divisionRule;
}

// generate powerset for a set of integer elts
vector<set<int>> powerSet(const vector<int> &elts)
{
    if (elts.empty())
    {
        return vector<set<int>>(1, set<int>());
    }
    else
    {
        vector<set<int>> allofthem = powerSet(
            vector<int>(elts.begin() + 1, elts.end()));
        int elt = elts[0];
        const int n = allofthem.size();
        for (int i = 0; i < n; ++i)
        {
            const set<int> &s = allofthem[i];
            allofthem.push_back(s);
            allofthem.back().insert(elt);
        }
        return allofthem;
    }
}

void printSet(set<int> A)
{
    for (auto elt : A)
    {
        cout << elt << " , ";
    }
}
// can probably turn this into a generic template function taking in container
void printVec(vector<int> A)
{
    for (auto elt : A)
    {
        cout << elt << " , ";
    }
}

void printVec(vector<double> A)
{
    for (auto elt : A)
    {
        cout << elt << " , ";
    }
}

vector<vector<int>> CartesianProduct(vector<vector<int>> A, vector<vector<int>> B)
{
    vector<vector<int>> ret;
    for (auto &a : A)
    {
        for (auto &b : B)
        {
            vector<int> c = a; // hopefully copy by value!
            // concatenate two vectors c and b!
            c.insert(c.end(), b.begin(), b.end());
            ret.push_back(c);
        }
    }
    return ret;
}

// self product of A "repeat" number of times
vector<vector<int>> CartesianSelfProduct(vector<int> A, int repeat)
{

    vector<vector<int>> A_transformed;
    for (auto elt : A)
    {
        A_transformed.push_back(vector<int>({elt}));
    }
    vector<vector<int>> ret = A_transformed;
    for (int i = 0; i < repeat - 1; i++)
    {
        ret = CartesianProduct(ret, A_transformed);
    }
    return ret;
}

int selectAction(const VectorXd &probs, mt19937_64 &generator, uniform_real_distribution<double> &u)
{
    // Implemtation by Philip Thomas

    // double temp = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
    // int chosen = numProposals - 1;
    // for (int a = 0; a < numProposals; a++)
    // {
    //     sum += probs[a];
    //     if (temp <= sum)
    //     {
    //         chosen = a;
    //         break;
    //     }
    // }

    // Implementation by Yao Li
    int chosen = 0;
    double tmp_double = probs[0];
    double tmp = u(generator);
    while (tmp_double < tmp && chosen < probs.size() - 1)
    {
        chosen++;
        tmp_double += probs[chosen];
    }
    return chosen;
}

// get the index of the proposal within agent.proposalSpace
// https://www.geeksforgeeks.org/how-to-find-index-of-a-given-element-in-a-vector-in-cpp/
int getIndex(vector<Proposal> &proposals, const Proposal &proposal)
{
    auto it = find(proposals.begin(), proposals.end(), proposal);
    // found elt
    if (it != proposals.end())
    {
        return it - proposals.begin();
    }
    else
    {
        cout << "SOMETHING IS WRONG CAN'T GET INDEX!!" << endl;
        return -1;
    }
}

// https://www.geeksforgeeks.org/counting-inversions/
int getInvCount(int arr[], int n)
{
    int inv_count = 0;
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j])
                inv_count++;

    return inv_count;
}
