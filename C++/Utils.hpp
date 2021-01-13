#pragma once
#include <Eigen/Dense>
#include "simplex_grid.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <tuple>
#include <random>
using namespace Eigen;
using namespace std;
// https://stackoverflow.com/questions/8833938/is-the-stdset-iteration-order-always-ascending-according-to-the-c-specificat#:~:text=Per%20the%20C%2B%2B%20standard%2C%20iteration,optional%20comparison%20predicate%20template%20argument.
// iterating through set always maintain the same order (non-descending order)
// which is very nice!
typedef set<int> Coalition;    // list of agentNames
typedef MatrixXd divisionRule; // each row is a share vector. This matrix
// rows consist of all regular grid points on the simplex.
// Shares are incremented in step of 10%. For example, if we have 2 players then
// divisionRule =
//  0   1
// 0.1 0.9
// 0.2 0.8
// 0.3 0.7
// 0.4 0.6
// 0.5 0.5
// 0.6 0.4
// 0.7 0.3
// 0.8 0.2
// 0.9 0.1
//   1   0
typedef pair<Coalition, VectorXd> Proposal;
// list of pair (coalition, divisionRule, coalitionValue
typedef vector<tuple<Coalition, VectorXd, double>> CoalitionStructure;

//MatrixXd softmax(const MatrixXd &M);

VectorXd softmax(const VectorXd &v);

MatrixXd generateDivisionRule(int numPlayers);

// generate powerset for a set of integer elts
vector<set<int>> powerSet(const vector<int> &elts);

void printSet(set<int> A);
void printVec(vector<int> A);
void printVec(vector<double> A);

vector<vector<int>> CartesianProduct(vector<vector<int>> A, vector<vector<int>> B);

vector<vector<int>> CartesianSelfProduct(vector<int> A, int repeat);
// select an index based on probabilities given in probs. Probs should
// already be a proper probability vector
int selectAction(const VectorXd &probs, mt19937_64 &generator, uniform_real_distribution<double> &u);

// get the index of the proposal within proposals
int getIndex(vector<Proposal> &proposals, const Proposal &proposal);
int getInvCount(int arr[], int n);

// http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
// Python-like enumerate in C++17!!
// need C++17 with at least g++-7 compiler!
template <typename T,
          typename TIter = decltype(std::begin(std::declval<T>())),
          typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T &&iterable)
{
    struct iterator
    {
        size_t i;
        TIter iter;
        bool operator!=(const iterator &other) const { return iter != other.iter; }
        void operator++()
        {
            ++i;
            ++iter;
        }
        auto operator*() const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper
    {
        T iterable;
        auto begin() { return iterator{0, std::begin(iterable)}; }
        auto end() { return iterator{0, std::end(iterable)}; }
    };
    return iterable_wrapper{std::forward<T>(iterable)};
}

// https://stackoverflow.com/questions/37368787/c-sort-one-vector-based-on-another-one
// to sort the division rule still remembering the name of the agents involved!
// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes
// that the vectors have equal length)

// https://stackoverflow.com/questions/18973042/symbols-not-found-for-architecture-x86-64-on-qtcreator-project
// Also, note that we cannot split template class/function into .h and .cpp parts!
// populate zipped
template <typename A, typename B>
void zip(
    pair<Coalition, VectorXd> proposal,
    std::vector<std::pair<A, B>> &zipped)
{

    Coalition &a = proposal.first;
    VectorXd &b = proposal.second;
    for (auto [i, agentName] : enumerate(a))
    {
        zipped.push_back(std::make_pair(agentName, b[i]));
    }
}

// populate a and b
template <typename A, typename B>
void unzip(
    const std::vector<std::pair<A, B>> &zipped,
    VectorXd &a,
    VectorXd &b)
{

    for (size_t i = 0; i < a.size(); i++)
    {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}

template <typename a>
vector<a> set2vec(set<a> s)
{
    std::vector<a> v(s.size());
    std::copy(s.begin(), s.end(), v.begin());
    return v;
}