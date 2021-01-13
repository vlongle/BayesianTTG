#pragma once
#include "Utils.hpp"
#include "Game.hpp"

class Algo
{
public:
    Algo(Game &game, mt19937_64 &generator) : game(game), generator(generator){};
    // pure virtual functions: child classes must implement these
    // this function should set changedBelief to True for all agents in non-singleton
    // coalitions where we update their beliefs!
    virtual void updateBelief(vector<Coalition> &nonSingletonCoals) = 0;
    // second return is a vector of non-singleton coalitions. The number of such coalitions
    // is either 0 or 1
    virtual pair<CoalitionStructure, vector<Coalition>> formationProcess() = 0;
    Game &game;
    mt19937_64 &generator;

    // use the public wealth signal to update agents' belief on top of the outcome-based
    // updating. This should be a no-op for bandit
    virtual void exploitSignal(double trustLevel) =0;
};
