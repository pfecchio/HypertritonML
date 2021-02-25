#include <TChain.h>
#include <TF1.h>
#include <TFile.h>
#include <TRandom3.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"

#include <vector>

#include "../../common/GenerateTable/Common.h"
#include "../../common/GenerateTable/GenTable2.h"
#include "../../common/GenerateTable/Table2.h"

void GenerateBkgTableFromMC() {

  string hypDataDir  = getenv("HYPERML_TREES__2");
  string hypTableDir = getenv("HYPERML_TABLES_2");

  string mcName = getenv("HYPERML_MC");

  string inFileName = "HyperTritonTree_" + mcName + ".root";
  string inFileArg  = hypDataDir + "/" + inFileName;

  string outFileName = "BkgTable.root";
  string outFileArg  = hypTableDir + "/" + outFileName;

  TChain mcChain("_default/fTreeV0");
  mcChain.AddFile(inFileArg.data());

  TTreeReader fReader(&mcChain);
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderValue<RCollision> RColl            = {fReader, "RCollision"};

  TFile outFile(outFileArg.data(), "RECREATE");
  Table2 outputTable("BkgTable", "Bkg table");

  while (fReader.Next()) {
    std::vector<float> index;
    for (auto &SHyper : SHyperVec) {
      int ind = SHyper.fRecoIndex;
      if (ind >= 0) {
        index.push_back(ind);
      }
    }

    float Rind = 0;
    for (auto &RHyper : RHyperVec) {
      bool fake = false;
      for (auto &i : index) {
        if (Rind == i) fake = true;
      }
      if (fake == false) outputTable.Fill(RHyper, *RColl, false);
      Rind = Rind + 1;
    }
  }
  outFile.cd();
  outputTable.Write();

  outFile.Close();
}
