#include <iostream>
#include <vector>
#include <exception>

#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TRandom3.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include "../../common/GenerateTable/O2Table3.h"

#include "AliAnalysisTaskHypertriton3.h"

// r data path "~/data/3body_hypertriton_data/O2/HyperTritonTree3_18r.root"
// q data path "~/data/3body_hypertriton_data/O2/HyperTritonTree3_18q.root"
// mc data path "~/data/3body_hypertriton_data/O2/SignalTable.root"

// output r  "~/data/3body_hypertriton_data/O2/DataTable_18r.root"
// output q  "~/data/3body_hypertriton_data/O2/DataTable_18q.root"
// output mc "~/data/3body_hypertriton_data/O2/SignalTableReweight.root"

void GenerateTableO2(std::string dataDir = "~/data/3body_hypertriton_data/O2/last_trees/HyperTritonTree_19d2.root", std::string tableDir = "~/data/3body_hypertriton_data/O2/SignalTableReweight.root", bool mc = true)
{
  TChain inputChain("Hyp3O2");
  inputChain.Add(dataDir.data());

  TFile outFile(tableDir.data(), "RECREATE");
  TableO2 tree(mc);

  TTreeReader fReader(&inputChain);

  if (mc) {
    TTreeReaderValue<SHyperTriton3O2> SHyper{fReader, "SHyperTriton"};

    // get the blast wave functions for the pt reshaping
    std::string hypUtilsDir = getenv("HYPERML_UTILS");
    std::string bwFileArg = hypUtilsDir + "/BlastWaveFits.root";

    // get the bw functions for the pt rejection
    TFile bwFile(bwFileArg.data());

    TF1 *BlastWave0{(TF1 *)bwFile.Get("BlastWave/BlastWave0")};
    TF1 *BlastWave1{(TF1 *)bwFile.Get("BlastWave/BlastWave1")};
    TF1 *BlastWave2{(TF1 *)bwFile.Get("BlastWave/BlastWave2")};

    double max0 = BlastWave0->GetMaximum();
    double max1 = BlastWave1->GetMaximum();
    double max2 = BlastWave2->GetMaximum();

    TF1 *BlastWaveArray[3]{BlastWave0, BlastWave1, BlastWave2};
    double MaxArray[3]{max0, max1, max2};

    while (fReader.Next()) {
      tree.Fill(*SHyper, BlastWaveArray, MaxArray);
    }

    outFile.cd();
    tree.Write();
    outFile.Close();

    std::cout << "\nDerived tables from MC generated!\n" << std::endl;
  } else {
    TTreeReaderValue<RHyperTriton3O2> RHyper{fReader, "RHyperTriton"};

    while (fReader.Next()) {
      tree.Fill(*RHyper);
    }

    outFile.cd();
    tree.Write();
    outFile.Close();

    std::cout << "\nDerived tables from Data generated!\n" << std::endl;
  }
}