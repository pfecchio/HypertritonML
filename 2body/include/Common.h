#ifndef HYPERCOMMON_H
#define HYPERCOMMON_H

#include <cmath>
#include <vector>
#include <stdlib.h>

#include <TColor.h>
#include <TF1.h>
#include <TLorentzVector.h>

constexpr double kHyperTritonMass{2.99131};
constexpr char kLetter[2]{'A','M'};

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

/// custom colors
const int kBlueC     = TColor::GetColor("#1f78b4");
const int kBlueCT    = TColor::GetColorTransparent(kBlueC, 0.5);
const int kRedC      = TColor::GetColor("#e31a1c");
const int kRedCT     = TColor::GetColorTransparent(kRedC, 0.5);
const int kPurpleC   = TColor::GetColor("#911eb4");
const int kPurpleCT  = TColor::GetColorTransparent(kPurpleC, 0.5);
const int kOrangeC   = TColor::GetColor("#ff7f00");
const int kOrangeCT  = TColor::GetColorTransparent(kOrangeC, 0.5);
const int kGreenC    = TColor::GetColor("#33a02c");
const int kGreenCT   = TColor::GetColorTransparent(kGreenC, 0.5);
const int kMagentaC  = TColor::GetColor("#f032e6");
const int kMagentaCT = TColor::GetColorTransparent(kMagentaC, 0.5);
const int kYellowC   = TColor::GetColor("#ffe119");
const int kYellowCT  = TColor::GetColorTransparent(kYellowC, 0.5);
const int kBrownC    = TColor::GetColor("#b15928");
const int kBrownCT   = TColor::GetColorTransparent(kBrownC, 0.5);

#endif
