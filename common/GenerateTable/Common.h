#ifndef HYPERCOMMON_H
#define HYPERCOMMON_H

#include <cmath>
#include <stdlib.h>
#include <vector>

#include <TColor.h>
#include <TF1.h>
#include <TLorentzVector.h>

constexpr double kHyperMass{2.99131};

constexpr float kDeuMass{1.87561};
constexpr float kPMass{0.938272};
constexpr float kPiMass{0.13957};

constexpr char kLetter[2]{'A', 'M'};

struct SumTF1 {

  SumTF1(const std::vector<TF1 *> &flist) : fFuncList(flist) {}

  double operator()(const double *x, const double *p) {
    double result = 0;
    for (unsigned int i = 0; i < fFuncList.size(); ++i)
      result += fFuncList[i]->EvalPar(x, p);
    return result;
  }

  std::vector<TF1 *> fFuncList;
};

inline float SProd(const TLorentzVector &a1, const TLorentzVector &a2) { return a1.Vect() * a2.Vect(); }

inline float Sq(float a) { return a * a; }

inline float VProd(const TLorentzVector &a1, const TLorentzVector &a2) {
  float x = a1[1] * a2[2] - a2[1] * a1[2];
  float y = a1[2] * a2[0] - a1[0] * a2[2];
  float z = a1[0] * a2[1] - a1[1] * a2[0];
  return std::sqrt(x * x + y * y + z * z);
}

template <typename T> double Pot2(T a) { return a * a; }

template <typename F> double Hypote(F a, F b) { return std::sqrt(a * a + b * b); }

template <typename F> double Hypote(F a, F b, F c) { return std::sqrt(a * a + b * b + c * c); }

template <typename F> double Hypote(F a, F b, F c, F d) { return std::sqrt(a * a + b * b + c * c + d * d); }

template <typename T> double DistanceZ(T v1, T v2) { return std::sqrt(Pot2(v1[2] - v2[2])); }

template <typename T> double DistanceXY(T v1, T v2) { return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1])); }

template <typename T> double Distance3D(T v1, T v2) {
  return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1]) + Pot2(v1[2] - v2[2]));
}

inline void getDirs2(char *&dataDir, char *&tableDir) {
  dataDir  = getenv("HYPERML_TREES__2");
  tableDir = getenv("HYPERML_TABLES_2");
  dataDir  = dataDir == NULL ? new char[2]{'.'} : dataDir;
  tableDir = tableDir == NULL ? new char[2]{'.'} : tableDir;
}

inline void getDirs3(char *&dataDir, char *&tableDir) {
  dataDir  = getenv("HYPERML_TREES__3");
  tableDir = getenv("HYPERML_TABLES_3");
  dataDir  = dataDir == NULL ? new char[2]{'.'} : dataDir;
  tableDir = tableDir == NULL ? new char[2]{'.'} : tableDir;
}

// get nClsITS from cluster map
int GetNClsITS(unsigned char clsMap) {
  int ncls = 0;
  for (int i = 0; i < 6; i++) {
    ncls += (int)(clsMap >> i) & 1;
  }
  return ncls;
}

/// custom colors
const int kBlueCustom     = TColor::GetColor("#1f78b4");
const int kBlueCustomT    = TColor::GetColorTransparent(kBlueCustom, 0.5);
const int kRedCustom      = TColor::GetColor("#e31a1c");
const int kRedCustomT     = TColor::GetColorTransparent(kRedCustom, 0.5);
const int kPurpleCustom   = TColor::GetColor("#911eb4");
const int kPurpleCustomT  = TColor::GetColorTransparent(kPurpleCustom, 0.5);
const int kOrangeCustom   = TColor::GetColor("#ff7f00");
const int kOrangeCustomT  = TColor::GetColorTransparent(kOrangeCustom, 0.5);
const int kGreenCustom    = TColor::GetColor("#33a02c");
const int kGreenCustomT   = TColor::GetColorTransparent(kGreenCustom, 0.5);
const int kMagentaCustom  = TColor::GetColor("#f032e6");
const int kMagentaCustomT = TColor::GetColorTransparent(kMagentaCustom, 0.5);
const int kYellowCustom   = TColor::GetColor("#ffe119");
const int kYellowCustomT  = TColor::GetColorTransparent(kYellowCustom, 0.5);
const int kBrownCustom    = TColor::GetColor("#b15928");
const int kBrownCustomT   = TColor::GetColorTransparent(kBrownCustom, 0.5);

#endif
