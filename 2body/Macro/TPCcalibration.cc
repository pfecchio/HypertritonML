#include <iostream>

#include <TF1.h>
#include <TFile.h>
#include <TH2D.h>
#include <TObjArray.h>

void TPCcalibration() {
  string hypTableDir = getenv("HYPERML_TABLES_2");
  string hypUtilsDir = getenv("HYPERML_UTILS");

  string inFileArg = hypTableDir + "/SignalTable.root";
  string outFileArg = hypUtilsDir + "/He3TPCCalibration.root";

  TFile *inFile = new TFile(inFileArg.data(), "READ");

  TH2D *hNSigmaTPCVsPtHe3 = dynamic_cast<TH2D *>(inFile->Get("nSigmaTPCvsPTHe3"));
  hNSigmaTPCVsPtHe3->SetDirectory(0);

  TF1 *gauss = new TF1("gauss", "gaus(0)", -8, 8);

  TObjArray aSlices;
  hNSigmaTPCVsPtHe3->FitSlicesY(gauss, 0, -1, 0, "R",&aSlices);

  TH1D *hCalibration = dynamic_cast<TH1D *>(aSlices.FindObject("nSigmaTPCvsPTHe3_1"));

  TF1 *fCalibFunction = new TF1("He3TPCCalib", "pol3(0)", 2, 10);
  hCalibration->Fit(fCalibFunction, "R", "", 2., 9.25);

  hCalibration->SetLineColor(kBlue);
  hCalibration->SetLineWidth(2);

  TFile outFile(outFileArg.data(), "RECREATE");

  hCalibration->Write();
  fCalibFunction->Write();

  outFile.Write();
  outFile.Close();
}