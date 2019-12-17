#include <TAxis.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TPad.h>
#include <TPaveText.h>
#include <TString.h>
#include <stdlib.h>

#include <string>
#include <vector>

using std::string;
using std::vector;

void canvas_partition(TCanvas *C, const Int_t Nx = 2, const Int_t Ny = 2, Float_t lMargin = 0.15,
                      Float_t rMargin = 0.05, Float_t bMargin = 0.15, Float_t tMargin = 0.05);
TH1F* peak_plot_makeup(string histo);

void peak_plot() {

  string hist12   = "0-90/pol2/ct24_pT210_cen090_eff0.71_pol2";
  string hist24   = "0-90/pol2/ct24_pT210_cen090_eff0.79_pol2";
  string hist46   = "0-90/pol2/ct24_pT210_cen090_eff0.79_pol2";
  string hist68   = "0-90/pol2/ct24_pT210_cen090_eff0.78_pol2";
  string hist810  = "0-90/pol2/ct24_pT210_cen090_eff0.81_pol2";
  string hist1014 = "0-90/pol2/ct24_pT210_cen090_eff0.79_pol2";
  string hist1418 = "0-90/pol2/ct24_pT210_cen090_eff0.74_pol2";
  string hist1823 = "0-90/pol2/ct24_pT210_cen090_eff0.71_pol2";
  string hist2335 = "0-90/pol2/ct24_pT210_cen090_eff0.61_pol2";

  vector<string> path_vector;
  vector<TH1F*> hist_vector;


  path_vector.push_back(hist12);
  path_vector.push_back(hist24);
  path_vector.push_back(hist46);
  path_vector.push_back(hist68);
  path_vector.push_back(hist810);
  path_vector.push_back(hist1014);
  path_vector.push_back(hist1418);
  path_vector.push_back(hist1823);
  path_vector.push_back(hist2335);

  for (auto h : path_vector) {
    auto hist = peak_plot_makeup(h);
    hist_vector.push_back(hist);
  }

  // TCanvas *c = new TCanvas("C", "canvas", 1024, 640);
  // c->SetFillStyle(4000);

  // // Number of PADS
  // const Int_t Nx = 3;
  // const Int_t Ny = 3;
  // // Margins
  // Float_t lMargin = 0.12;
  // Float_t rMargin = 0.05;
  // Float_t bMargin = 0.15;
  // Float_t tMargin = 0.05;
  // // Canvas setup
  // canvas_partition(c, Nx, Ny, lMargin, rMargin, bMargin, tMargin);

  // TPad *pad[Nx][Ny];

  // for (Int_t i = 0; i < Nx; i++) {
  //   for (Int_t j = 0; j < Ny; j++) {
  //     c->cd(0);
  //     // Get the pads previously created.
  //     char pname[16];
  //     pad[i][j] = (TPad *)gROOT->FindObject(pname);
  //     pad[i][j]->Draw();
  //     pad[i][j]->SetFillStyle(4000);
  //     pad[i][j]->SetFrameFillStyle(4000);
  //     pad[i][j]->cd();
  //     // Size factors
  //     Float_t xFactor = pad[0][0]->GetAbsWNDC() / pad[i][j]->GetAbsWNDC();
  //     Float_t yFactor = pad[0][0]->GetAbsHNDC() / pad[i][j]->GetAbsHNDC();
  //     char hname[16];
  //     TH1F *hFrame = (TH1F *)h->Clone(hname);
  //     hFrame->Reset();
  //     hFrame->Draw();
  //     // y axis range
  //     hFrame->GetYaxis()->SetRangeUser(0.0001, 1.2 * h->GetMaximum());
  //     // Format for y axis
  //     hFrame->GetYaxis()->SetLabelFont(43);
  //     hFrame->GetYaxis()->SetLabelSize(16);
  //     hFrame->GetYaxis()->SetLabelOffset(0.02);
  //     hFrame->GetYaxis()->SetTitleFont(43);
  //     hFrame->GetYaxis()->SetTitleSize(16);
  //     hFrame->GetYaxis()->SetTitleOffset(5);
  //     hFrame->GetYaxis()->CenterTitle();
  //     hFrame->GetYaxis()->SetNdivisions(505);
  //     // TICKS Y Axis
  //     hFrame->GetYaxis()->SetTickLength(xFactor * 0.04 / yFactor);
  //     // Format for x axis
  //     hFrame->GetXaxis()->SetLabelFont(43);
  //     hFrame->GetXaxis()->SetLabelSize(16);
  //     hFrame->GetXaxis()->SetLabelOffset(0.02);
  //     hFrame->GetXaxis()->SetTitleFont(43);
  //     hFrame->GetXaxis()->SetTitleSize(16);
  //     hFrame->GetXaxis()->SetTitleOffset(5);
  //     hFrame->GetXaxis()->CenterTitle();
  //     hFrame->GetXaxis()->SetNdivisions(505);
  //     // TICKS X Axis
  //     hFrame->GetXaxis()->SetTickLength(yFactor * 0.06 / xFactor);
  //     h->Draw("same");
  //   }
  // }
  // c->cd();
}

void peak_plot_makeup(string histo) {
  // custom colors
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

  TF1 *sigmaParam = new TF1("sigmaParam", "pol2", 0, 35);
  sigmaParam->SetParameters(1.544e-3, 7.015e-5, -1.965e-6);

  std::string inDir  = getenv("HYPERML_RESULTS_2");
  std::string inName = "/ct_analysis_results.root";
  inDir.append(inName);

  TFile fInput(inDir.data(), "READ");

  TH1D *hInvMass = dynamic_cast<TH1D *>(fInput.Get(histo.data()));

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

  TPaveText paveText(0.55, 0.65, 0.95, 0.85, "NDC");
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

  auto legend = new TLegend(0.16, 0.64, 0.52, 0.84);
  legend->AddEntry(hInvMass, "Data", "pe");
  legend->AddEntry(fFit, "Signal + Background", "l");
  legend->AddEntry(fGauss, "Signal", "l");
  legend->AddEntry(fPol2, "Background", "l");

  TCanvas c("cInvMass", "");

  auto frame = gPad->DrawFrame(2.96, 0.001, 3.041, 125,
                               ";#it{M} (^{3}He + #pi^{-}) (GeV/#it{c}^{2});Counts / (2.25 MeV/#it{c}^{2})");
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

  return frame;

  // std::string outDir     = getenv("HYPERML_FIGURES_2");
  // std::string outNamePdf = "/peak24_plot.pdf";
  // std::string outNameEps = "/peak24_plot.eps";

  // c.SaveAs((outDir + outNamePdf).data());
  // c.SaveAs((outDir + outNameEps).data());

  // TFile fOutput("InvMass.root", "RECREATE");

  // c.Write();
  // fOutput.Close();
}

void canvas_partition(TCanvas *C, const Int_t Nx, const Int_t Ny, Float_t lMargin, Float_t rMargin, Float_t bMargin,
                      Float_t tMargin) {
  if (!C) return;
  // Setup Pad layout:
  Float_t vSpacing = 0.0;
  Float_t vStep    = (1. - bMargin - tMargin - (Ny - 1) * vSpacing) / Ny;
  Float_t hSpacing = 0.0;
  Float_t hStep    = (1. - lMargin - rMargin - (Nx - 1) * hSpacing) / Nx;
  Float_t vposd, vposu, vmard, vmaru, vfactor;
  Float_t hposl, hposr, hmarl, hmarr, hfactor;
  for (Int_t i = 0; i < Nx; i++) {
    if (i == 0) {
      hposl   = 0.0;
      hposr   = lMargin + hStep;
      hfactor = hposr - hposl;
      hmarl   = lMargin / hfactor;
      hmarr   = 0.0;
    } else if (i == Nx - 1) {
      hposl   = hposr + hSpacing;
      hposr   = hposl + hStep + rMargin;
      hfactor = hposr - hposl;
      hmarl   = 0.0;
      hmarr   = rMargin / (hposr - hposl);
    } else {
      hposl   = hposr + hSpacing;
      hposr   = hposl + hStep;
      hfactor = hposr - hposl;
      hmarl   = 0.0;
      hmarr   = 0.0;
    }
    for (Int_t j = 0; j < Ny; j++) {
      if (j == 0) {
        vposd   = 0.0;
        vposu   = bMargin + vStep;
        vfactor = vposu - vposd;
        vmard   = bMargin / vfactor;
        vmaru   = 0.0;
      } else if (j == Ny - 1) {
        vposd   = vposu + vSpacing;
        vposu   = vposd + vStep + tMargin;
        vfactor = vposu - vposd;
        vmard   = 0.0;
        vmaru   = tMargin / (vposu - vposd);
      } else {
        vposd   = vposu + vSpacing;
        vposu   = vposd + vStep;
        vfactor = vposu - vposd;
        vmard   = 0.0;
        vmaru   = 0.0;
      }
      C->cd(0);
      char name[16];
      TPad *pad = (TPad *)gROOT->FindObject(name);
      if (pad) delete pad;
      pad = new TPad(name, "", hposl, vposd, hposr, vposu);
      pad->SetLeftMargin(hmarl);
      pad->SetRightMargin(hmarr);
      pad->SetBottomMargin(vmard);
      pad->SetTopMargin(vmaru);
      pad->SetFrameBorderMode(0);
      pad->SetBorderMode(0);
      pad->SetBorderSize(0);
      pad->Draw();
    }
  }
}