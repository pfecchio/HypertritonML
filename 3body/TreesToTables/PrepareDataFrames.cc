#include <vector>
#include <TCanvas.h>
#include <TRandom3.h>
#include <ROOT/RDataFrame.hxx>
#include <TF1.h>

#include <string>
#include <fstream>

using namespace ROOT;
using namespace TMVA::Experimental;

void PrepareDataFrames(std::string dType = "data", std::string hypDataDir = "", std::string hypTableDir = "") {

  ROOT::EnableImplicitMT();

  if (hypDataDir == "")
    hypDataDir = getenv("HYPERML_DATA_3");
  if (hypTableDir == "")
    hypTableDir = getenv("HYPERML_TABLES_3");
  
  
  std::ifstream in(Form("%s/%s_path_list", hypDataDir.data(), dType.data()));

  std::string outFile = hypTableDir + "/" + "HypDataTable.root";

  std::vector<std::string> vecOfStrs;
  std::string str;

  while (std::getline(in, str))
  {
    if(str.size() > 0)
      vecOfStrs.push_back(hypDataDir + "/" + str);
  }

  ROOT::RDataFrame df("Hyp3O2",vecOfStrs);
  // managing Double_32t variables + renaming
  std::vector<std::string> inFeatures{"dca_de","dca_pr","dca_pi","dca_de_sv","dca_pr_sv","dca_pi_sv","tpcClus_de","tpcClus_pr","tpcClus_pi","tpcNsig_de","tpcNsig_pr","tpcNsig_pi","dca_de_pr","dca_de_pi","dca_pr_pi","mppi_vert","mppi","mdpi","momDstar","cosTheta_ProtonPiH","cosThetaStar","cosPA"};
  std::vector<std::string> outFeatures{"dca_de_f","dca_pr_f","dca_pi_f","dca_de_sv_f","dca_pr_sv_f","dca_pi_sv_f","tpc_ncls_de_f","tpc_ncls_pr_f","tpc_ncls_pi_f","tpc_nsig_de_f","tpc_nsig_pr_f","tpc_nsig_pi_f","dca_de_pr_f","dca_de_pi_f","dca_pr_pi_f","mppi_vert_f","mppi_f","mdpi_f","mom_dstar_f","cos_theta_ppi_H_f","cos_theta_star_f","cos_pa_f"};

  auto filterDF = df.Filter("cosPA > 0 ");

  for (int i=0; i < inFeatures.size(); i++) {
    filterDF = filterDF.Define(outFeatures[i], Form("(float)%s", inFeatures[i].data()));
  }

  outFeatures.push_back("positive");
  outFeatures.push_back("pt");
  outFeatures.push_back("pz");
  outFeatures.push_back("r");
  outFeatures.push_back("m");
  outFeatures.push_back("ct");
  outFeatures.push_back("chi2");
  outFeatures.push_back("centrality");
  filterDF.Snapshot("DataTable", outFile, outFeatures);

}
