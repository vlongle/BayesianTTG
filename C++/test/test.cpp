#include <iostream>
#include <set>
#include <vector>
#include "Eigen/Dense"
#include <fstream>
using namespace std;
using namespace Eigen;


#include <tuple>

template <typename T,
                   typename TIter = decltype(std::begin(std::declval<T>())),
                             typename = decltype(std::end(std::declval<T>()))>
                             constexpr auto enumerate(T && iterable)
{
        struct iterator
        {
                    size_t i;
                            TIter iter;
                                    bool operator != (const iterator & other) const { return iter != other.iter;  }
                                            void operator ++ () { ++i; ++iter;  }
                                                    auto operator * () const { return std::tie(i, *iter);  }
                                                        
        };
            struct iterable_wrapper
            {
                        T iterable;
                                auto begin() { return iterator{ 0, std::begin(iterable)  };  }
                                        auto end() { return iterator{ 0, std::end(iterable)  };  }
                                            
            };
                return iterable_wrapper{ std::forward<T>(iterable)  };

}


class Base{
    public:
        void outcome(){
            cout << "base outcome" << endl;
        }
        void formation(){
            cout << "Base formation " << endl;
            outcome();
        }
};
class Derived: public Base{
        void outcome(){
            cout << "derived outcome" << endl;
        }
};


template <typename a>
int countInversions(vector<a> groundTruth, vector<a> candidate){
    if (groundTruth.size() != candidate.size()){
                cout << "CRITICAL!!!!! Something wrong in countInversion() function" << endl;
                        return 0;
                            
    }

        int n = groundTruth.size();
            int inversions = 0;
            for (int i=0; i < n-1; i++){
                for (int j=i+1; j<n; j++){

                    if (groundTruth[i] > groundTruth[j] && candidate[i] <= candidate[j]){
                                        inversions++;
                                                    
                    }
                    else if (groundTruth[i] < groundTruth[j] && candidate[i] >= candidate[j]){
                                        inversions++;
                                                    
                    }
                    else if (groundTruth[i] == groundTruth[j] && candidate[i] != candidate[j]){
                                        inversions++;
                                                    
                    }
                            
                }
                    
            }

                return inversions;

}

//int main(){
//    VectorXd v(2);
//    v << 1,2;
//    cout << v << endl;
//
//    ofstream out("test.txt");
//    out << v;
//        out.close();
//}
