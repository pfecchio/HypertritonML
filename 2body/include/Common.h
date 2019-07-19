#ifndef HYPERCOMMON_H
#define HYPERCOMMON_H

#include <cmath>
#include <vector>
#include <stdlib.h>

#include <TF1.h>
#include <TLorentzVector.h>

struct SumTF1
{

    SumTF1(const std::vector<TF1 *> &flist) : fFuncList(flist) {}

    double operator()(const double *x, const double *p)
    {
        double result = 0;
        for (unsigned int i = 0; i < fFuncList.size(); ++i)
            result += fFuncList[i]->EvalPar(x, p);
        return result;
    }

    std::vector<TF1 *> fFuncList;
};

inline float SProd(const TLorentzVector &a1, const TLorentzVector &a2)
{

    return a1.Vect() * a2.Vect();
}

inline float Sq(float a)
{
    return a * a;
}

inline float VProd(const TLorentzVector &a1, const TLorentzVector &a2)
{
    float x = a1[1] * a2[2] - a2[1] * a1[2];
    float y = a1[2] * a2[0] - a1[0] * a2[2];
    float z = a1[0] * a2[1] - a1[1] * a2[0];
    return std::sqrt(x * x + y * y + z * z);
}

template <typename F>
double Hypot(F a, F b)
{
    return std::sqrt(a * a + b * b);
}

template <typename F>
double Hypot(F a, F b, F c)
{
    return std::sqrt(a * a + b * b + c * c);
}

template <typename F>
double Hypot(F a, F b, F c, F d)
{
    return std::sqrt(a * a + b * b + c * c + d * d);
}

inline void getDirs(char* &dataDir,char* &tableDir) {
  dataDir =  getenv("HYPERML_DATA");
  tableDir = getenv("HYPERML_TABLES");
  dataDir = dataDir == NULL ? new char[2]{'.'} : dataDir;
  tableDir = tableDir == NULL ? new char[2]{'.'} : tableDir;
}

#endif
