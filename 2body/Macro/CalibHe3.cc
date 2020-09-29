#include <cmath>
#include <string>
#include "AliExternalTrackParam.h"
#include "ROOT/RDataFrame.hxx"
#include "TDirectory.h"
#include "TF1.h"
#include "TH1D.h"

double BetheBlochAleph(double *x, double *p) {
  return AliExternalTrackParam::BetheBlochAleph(x[0] / 2.80923f, p[0], p[1], p[2], p[3], p[4]);
}

void CalibHe3(std::string tablefile) {
  ROOT::RDataFrame rdf("DataTable",tablefile); 
  auto h2 = rdf.Filter("NpidClustersHe3 > 100").Histo2D({"dedx",";TPC #it{p} (GeV/#it{c});dE/dx (a.u.);",18,1.4,5,256,0,2048},"TPCmomHe3","TPCsignalHe3");
  h2->FitSlicesY();
  TH1* mean = (TH1*)gDirectory->Get("dedx_1");
  TH1* sigma = (TH1*)gDirectory->Get("dedx_2");
  sigma->Divide(mean);
  sigma->Fit("pol0");

  TF1 mybethe("mybethe",BetheBlochAleph, 1, 6, 5);
  double starting_pars[5]{-166.11733,-0.11020473,0.10851357,2.7018593,-0.087597824};
  mybethe.SetParameters(starting_pars);

  for (int i{0}; i < 10; ++i)
    mean->Fit(&mybethe);
  mean->Draw();
}
