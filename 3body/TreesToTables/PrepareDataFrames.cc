#include <vector>
#include <TCanvas.h>
#include <TRandom3.h>
#include <ROOT/RDataFrame.hxx>
#include <TF1.h>

#include <string>
#include <fstream>

using namespace ROOT;
using namespace TMVA::Experimental;

void PrepareDataFrames() {

  ROOT::EnableImplicitMT();

  std::ifstream in("lists");
  std::vector<std::string> vecOfStrs;
  std::string str;
  while (std::getline(in, str))
  {
    if(str.size() > 0)
      vecOfStrs.push_back(str);
  }

  ROOT::RDataFrame df("Hyp3O2",vecOfStrs);
  std::vector<std::string> features{"dca_de","dca_pr","dca_pi","dca_de_sv","dca_pr_sv","dca_pi_sv","tpcClus_de","tpcClus_pr","tpcClus_pi","tpcNsig_de","tpcNsig_pr","tpcNsig_pi","dca_de_pr","dca_de_pi","dca_pr_pi","mppi_vert"};
  auto filterDF = df.Filter("cosPA > 0 ");
  for (auto& f : features) {
    filterDF = filterDF.Define(f + "_f", Form("(float)%s",f.data()));
    f += "_f";
  }
  features.push_back("cosPA");
  features.push_back("positive");
  features.push_back("pt");
  features.push_back("pz");
  features.push_back("r");
  features.push_back("m");
  features.push_back("ct");
  features.push_back("chi2");

  auto rHist = filterDF.Histo1D({"rHist",";r (cm); Entries", 200,0,200},"r");
  filterDF.Snapshot("DataTable","../../../tmp/HypTableDataLS.root",features);

  TFile output("output.root","recreate");
  TCanvas rCv("rCv");
  rHist->Draw();
  rCv.SaveAs("rCv.png");
  rHist->Write();
}