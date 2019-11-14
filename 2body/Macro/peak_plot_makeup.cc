#include <stdlib.h>
#include <TFile.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <TAxis.h>
#include <TPaveText.h>
#include <TString.h>
#include <TF1.h>
#include <TColor.h>
#include <TLegend.h>


void peak_plot_makeup() { 

  // custom colors
  const int kBlueC  = TColor::GetColor("#1f78b4");
  const int kBlueCT = TColor::GetColorTransparent(kBlueC, 0.5);
  const int kRedC  = TColor::GetColor("#e31a1c");
  const int kRedCT = TColor::GetColorTransparent(kRedC, 0.5);
  const int kPurpleC  = TColor::GetColor("#911eb4");
  const int kPurpleCT = TColor::GetColorTransparent(kPurpleC, 0.5);
  const int kOrangeC  = TColor::GetColor("#ff7f00");
  const int kOrangeCT = TColor::GetColorTransparent(kOrangeC, 0.5);
  const int kGreenC  = TColor::GetColor("#33a02c");
  const int kGreenCT = TColor::GetColorTransparent(kGreenC, 0.5);
  const int kMagentaC  = TColor::GetColor("#f032e6");
  const int kMagentaCT = TColor::GetColorTransparent(kMagentaC, 0.5);
  const int kYellowC  = TColor::GetColor("#ffe119");
  const int kYellowCT = TColor::GetColorTransparent(kYellowC, 0.5);
  const int kBrownC  = TColor::GetColor("#b15928");
  const int kBrownCT = TColor::GetColorTransparent(kBrownC, 0.5);

  TF1 *sigmaParam = new TF1("sigmaParam","pol2",0,35);
  sigmaParam->SetParameters(1.544e-3,7.015e-5,-1.965e-6);

  std::string inDir = getenv("HYPERML_RESULTS_2");
  std::string inName =  "/ct_analysis_results.root";
  inDir.append(inName);

  TFile fInput(inDir.data(), "READ");

  TH1D *hInvMass = dynamic_cast<TH1D*>(fInput.Get("0-90/pol2/ct24_pT210_cen090_eff0.79_pol2"));

  TF1 *fOldFit = hInvMass->GetFunction("fitTpl");
  fOldFit->Delete();

  hInvMass->SetStats(0);
  hInvMass->SetLineColor(kBlack);
  hInvMass->SetMarkerColor(kBlack);
  hInvMass->GetXaxis()->SetTitle("m (^{3}He + #pi)  (GeV/#it{c}^{2})");
  hInvMass->GetXaxis()->SetTitleSize(22);
  hInvMass->GetXaxis()->SetLabelSize(20);
  hInvMass->GetYaxis()->SetTitleSize(22);
  hInvMass->GetYaxis()->SetLabelSize(20);

  TPaveText paveText(0.55,0.65,0.95,0.85,"NDC");
  paveText.SetBorderSize(0);
  paveText.SetFillStyle(0);
  paveText.SetTextAlign(22);
  paveText.SetTextFont(43);
  paveText.SetTextSize(22);

  TString string1 = "#bf{ALICE Performance}";
  TString string2 = "Pb#minusPb  #sqrt{#it{s}_{NN}} = 5.02 TeV";
  TString string3 = "2 #leq  #it{c}t  < 4 cm,  0-90 % ";
  paveText.AddText(string1);
  paveText.AddText(string2);
  paveText.AddText(string3);

  TF1 *fFit = new TF1("fitF", "pol2(0)+gausn(3)", 0, 5);
  fFit->SetParName(0, "B_0");
  fFit->SetParName(1, "B_1");
  fFit->SetParName(2, "B_2");
  fFit->SetParName(3, "N_{sig}");
  fFit->SetParName(4, "#mu");
  fFit->SetParName(5, "#sigma");

  fFit->SetParameter(3, 40);
  fFit->SetParLimits(3, 0.001, 10000);
  fFit->SetParameter(4, 2.991);
  fFit->SetParLimits(4, 2.986, 3);
  fFit->FixParameter(5, sigmaParam->Eval(3));

  fFit->SetNpx(1000);
  fFit->SetLineWidth(2);
  fFit->SetLineColor(kBlueC);

  hInvMass->Fit(fFit, "QRL0+", "", 2.96, 3.05);
  hInvMass->SetDrawOption("e");

  TF1 *fGauss = new TF1("fitSig", "gausn(0)", 2.980, 3.003);
  fGauss->SetNpx(1000);
  fGauss->SetLineWidth(2);
  fGauss->SetLineStyle(2);
  fGauss->SetLineColor(kOrangeC);
  fGauss->SetParameter(0, fFit->GetParameter(3));
  fGauss->SetParameter(1, fFit->GetParameter(4));
  fGauss->SetParameter(2, fFit->GetParameter(5));

  TF1 *fPol2 = new TF1("fitBkg", "pol2(0)", 0, 5);
  fPol2->SetNpx(1000);
  fPol2->SetLineWidth(2);
  fPol2->SetLineStyle(2);
  fPol2->SetLineColor(kGreenC);
  fPol2->SetParameters(fFit->GetParameters());

  auto legend = new TLegend(0.16,0.64,0.52,0.84);
  legend->AddEntry(hInvMass, "Data","pe");
  legend->AddEntry(fFit, "Signal + Background","l");
  legend->AddEntry(fGauss, "Signal","l");
  legend->AddEntry(fPol2, "Background","l");

  TCanvas c("cInvMass", "");

  auto frame = gPad->DrawFrame(2.96, 0.001, 3.041, 125, ";#it{M} (^{3}He + #pi^{-}) (GeV/#it{c}^{2});Counts / (2.25 MeV/#it{c}^{2})");
  frame->GetYaxis()->SetTitleSize(26);
  frame->GetYaxis()->SetLabelSize(22);
  frame->GetXaxis()->SetTitleSize(26);
  frame->GetXaxis()->SetLabelSize(22);

  fGauss->Draw("same");
  fPol2->Draw("same");
  fFit->Draw("same");
  hInvMass->Draw("same");
  paveText.Draw();
  legend->Draw();

  std::string outDir = getenv("HYPERML_FIGURES_2");
  std::string outNamePdf = "/peak24_plot.pdf";
  std::string outNameEps = "/peak24_plot.eps";

  c.SaveAs((outDir + outNamePdf).data());
  c.SaveAs((outDir + outNameEps).data());

  // TFile fOutput("InvMass.root", "RECREATE");

  // c.Write();
  // fOutput.Close();
}