#ifndef GENERATE_DERIVED_TREE
#define GENERATE_DERIVED_TREE

template <typename T> double Pot2(T a) { return a * a; }

template <typename T> double Hypot(T a, T b, T c, T d) { return std::sqrt(Pot2(a) + Pot2(b) + Pot2(c) + Pot2(d)); }

template <typename T> double DistanceZ(T v1, T v2) { return std::sqrt(Pot2(v1[2] - v2[2])); }

template <typename T> double DistanceXY(T v1, T v2) { return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1])); }

template <typename T> double Distance3D(T v1, T v2) {
  return std::sqrt(Pot2(v1[0] - v2[0]) + Pot2(v1[1] - v2[1]) + Pot2(v1[2] - v2[2]));
}

int GetNClsITS(unsigned char clsMap) {
  int ncls = 0;

  for (int i = 0; i < 6; i++) {
    ncls += (int)(clsMap >> i) & 1;
  }

  return ncls;
}

#endif