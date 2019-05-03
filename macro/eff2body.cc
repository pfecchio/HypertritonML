#include <TFile.h>
#include <TLorentzVector.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <iostream>
#include <vector>

#include <TCanvas.h>
#include <TFile.h>
#include <TGaxis.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TLegend.h>
#include <TLegendEntry.h>
#include <TList.h>
#include <TMath.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TRatioPlot.h>
#include <TStyle.h>

#include "AliAnalysisTaskHyperTriton2He3piML.h"
#include "AliPID.h"

template <typename T> double Pot2(T a) { return a * a; }

template <typename T> double Hypot(T a, T b, T c, T d) { return std::sqrt(Pot2(a) + Pot2(b) + Pot2(c) + Pot2(d)); }

template <typename T> double Distance(T pX, T pY, T pZ, T dX, T dY, T dZ) {
  return std::sqrt(Pot2(pX - dX) + Pot2(pY - dY) + Pot2(pZ - dZ));
}

void histo_makeup(TH1 *h, int color, string xTitle = "", string yTitle = "", int lWidth = 2, int mSize = 1,
                  int mStyle = 0, string opt = "E", Bool_t stat = kFALSE);

void ratio_plot(TH1 *h1, TH1 *h2, TLegend *l, string mode, float lRange, float uRange, string cName = "c",
                int cXSize = 700, int cYSize = 500, int fColor = TColor::GetColor("#911eb4"));

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//       macro for compute hypertriton 2 body decay efficiency plots         //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

void eff2body() {

  /// custom colors
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

  const int colors[9] = {kBlack, kBlueC, kRedC, kPurpleC, kOrangeC, kGreenC, kMagentaC, kYellowC, kBrownC};

  const char lAM[3]{"AM"};
  const char *lCent[4]{"010", "1030", "3050", "1040"};

  //______________________________________________________________________________

  /// open input file and get the tree
  TFile *myFile         = TFile::Open("~/data/hyper2body_data/HyperTritonTree_lhc16h7abc_MCvert.root", "r");
  TDirectoryFile *mydir = (TDirectoryFile *)myFile->Get("_default");

  TTreeReader fReader("fTreeV0", mydir);
  TTreeReaderArray<SHyperTritonHe3pi> SHyperVec = {fReader, "SHyperTriton"};
  TTreeReaderArray<RHyperTritonHe3pi> RHyperVec = {fReader, "RHyperTriton"};
  TTreeReaderValue<RCollision> RColl            = {fReader, "RCollision"};

  TH1D *fHistGen[2];
  TH1D *fHistRec[2];

  TH1D *fHistGenCent[3];
  TH1D *fHistRecCent[3];

  TH1D *fHistGenCT;
  TH1D *fHistRecCT;

  for (int iMatter = 0; iMatter < 2; iMatter++) {
    fHistGen[iMatter] = new TH1D("fHistGen", "", 40, 0, 10);
    fHistGen[iMatter]->SetDirectory(0);
    fHistRec[iMatter] = new TH1D("fHistRec", "", 40, 0, 10);
    fHistRec[iMatter]->SetDirectory(0);
  }

  for (int iCent = 0; iCent < 4; iCent++) {
    fHistGenCent[iCent] = new TH1D(Form("fHistCentGen_%s", lCent[iCent]), "", 40, 0, 10);
    // fHistGenCent[iCent]->SetDirectory(0);
    fHistRecCent[iCent] = new TH1D(Form("fHistCentRec_%s", lCent[iCent]), "", 40, 0, 10);
    // fHistRecCent[iCent]->SetDirectory(0);
  }

  fHistGenCT = new TH1D("fHistGenCT", "", 20, 0, 40);
  fHistGenCT->SetDirectory(0);
  fHistRecCT = new TH1D("fHistRecCT", "", 20, 0, 40);
  fHistRecCT->SetDirectory(0);

  TH1D *fEfficiency[4];
  TH1D *fEfficiencyCent[4];
  TH1D *fEfficiencyCT;
  TH1D *fEfficiencyCentComp;

  fEfficiency[0] = new TH1D("fEfficiencyA", "", 40, 0, 10);
  fEfficiency[0]->SetDirectory(0);
  fEfficiency[1] = new TH1D("fEfficiencyM", "", 40, 0, 10);
  fEfficiency[1]->SetDirectory(0);
  fEfficiency[2] = new TH1D("fEfficiency_tot", "", 40, 0, 10);
  fEfficiency[2]->SetDirectory(0);

  fEfficiency[3] = new TH1D("fEfficiency_tot_1040", "", 40, 0, 10);
  fEfficiency[3]->SetDirectory(0);

  for (int iCent = 0; iCent < 4; iCent++) {
    fEfficiencyCent[iCent] = new TH1D(Form("fEffCent_%s", lCent[iCent]), "", 40, 0, 10);
    fEfficiencyCent[iCent]->SetDirectory(0);
  }

  fEfficiencyCT = new TH1D("fEfficiencyCT", "", 20, 0, 40);
  fEfficiencyCT->SetDirectory(0);

  const float bin[7]  = {0, 1, 2, 3, 4, 5, 9};
  fEfficiencyCentComp = new TH1D("fEfficiency1040", "", 6, bin);
  fEfficiencyCentComp->SetDirectory(0);

  TH2D *fHistDeltaCT;

  fHistDeltaCT = new TH2D("fHistDeltaCT", ";#Delta#it{c}t (cm) ;#it{c}t (cm)", 200, -5, 5, 20, 0, 40);
  fHistDeltaCT->SetDirectory(0);

  //------------------------------------------------------------
  // main loop on the tree
  //------------------------------------------------------------
  while (fReader.Next()) {
    auto cent = RColl->fCent;
    for (int i = 0; i < (static_cast<int>(SHyperVec.GetSize())); i++) {

      /// generated hypertritons
      auto sHyper = SHyperVec[i];
      auto reco   = sHyper.fRecoIndex;

      bool matter = sHyper.fPdgCode > 0;

      TLorentzVector sHe3, sPi, sMother;
      double eHe3_s = Hypot(sHyper.fPxHe3, sHyper.fPyHe3, sHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
      double ePi_s  = Hypot(sHyper.fPxPi, sHyper.fPyPi, sHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));

      sHe3.SetPxPyPzE(sHyper.fPxHe3, sHyper.fPyHe3, sHyper.fPzHe3, eHe3_s);
      sPi.SetPxPyPzE(sHyper.fPxPi, sHyper.fPyPi, sHyper.fPzPi, ePi_s);
      sMother = sHe3 + sPi;

      auto pt_gen = sMother.Pt();
      fHistGen[matter]->Fill(pt_gen);

      if (cent <= 10.0) fHistGenCent[0]->Fill(pt_gen);
      if (cent > 10.0 && cent <= 30.0) fHistGenCent[1]->Fill(pt_gen);
      if (cent > 30.0 && cent <= 50.0) fHistGenCent[2]->Fill(pt_gen);
      if (cent > 10.0 && cent <= 40.0) fHistGenCent[3]->Fill(pt_gen);

      /// compute the ct
      auto d_gen  = std::sqrt(Pot2(sHyper.fDecayX) + Pot2(sHyper.fDecayY) + Pot2(sHyper.fDecayZ));
      auto ct_gen = 2.992 * d_gen / sMother.P();
      fHistGenCT->Fill(ct_gen);

      if (sHyper.fFake) continue;

      /// reconstructed hypertritons
      auto rHyper = RHyperVec[reco];

      TLorentzVector rHe3, rPi, rMother;
      double eHe3_r = Hypot(rHyper.fPxHe3, rHyper.fPyHe3, rHyper.fPzHe3, AliPID::ParticleMass(AliPID::kHe3));
      double ePi_r  = Hypot(rHyper.fPxPi, rHyper.fPyPi, rHyper.fPzPi, AliPID::ParticleMass(AliPID::kPion));

      rHe3.SetPxPyPzE(rHyper.fPxHe3, rHyper.fPyHe3, rHyper.fPzHe3, eHe3_r);
      rPi.SetPxPyPzE(rHyper.fPxPi, rHyper.fPyPi, rHyper.fPzPi, ePi_r);
      rMother = rHe3 + rPi;

      auto pt_rec = rMother.Pt();
      fHistRec[matter]->Fill(pt_rec);

      if (cent <= 10.0) fHistRecCent[0]->Fill(pt_rec);
      if (cent > 10.0 && cent <= 30.0) fHistRecCent[1]->Fill(pt_rec);
      if (cent > 30.0 && cent <= 50.0) fHistRecCent[2]->Fill(pt_rec);
      if (cent > 10.0 && cent <= 40.0) fHistRecCent[3]->Fill(pt_rec);

      /// compute the ct
      auto d_rec  = std::sqrt(Pot2(rHyper.fDecayX) + Pot2(rHyper.fDecayY) + Pot2(rHyper.fDecayZ));
      auto ct_rec = 2.992 * d_rec / rMother.P();
      fHistRecCT->Fill(ct_rec);

      // fill the delta ct vs ct histo
      auto delta_ct = ct_gen - ct_rec;
      fHistDeltaCT->Fill(delta_ct, ct_gen);
    }
  }

  //------------------------------------------------------------
  // efficiency vs pT calculation
  //------------------------------------------------------------

  double counts_p[3][40];
  double ref_p[3][40];
  double eff_p[3][40];
  double err_p[3][40];

  for (int iMatter = 0; iMatter < 2; iMatter++) {
    for (int iPt = 0; iPt < 40; iPt++) {
      ref_p[iMatter][iPt]    = fHistGen[iMatter]->GetBinContent(iPt + 1);
      counts_p[iMatter][iPt] = fHistRec[iMatter]->GetBinContent(iPt + 1);

      eff_p[iMatter][iPt] = counts_p[iMatter][iPt] / ref_p[iMatter][iPt];
      err_p[iMatter][iPt] = std::sqrt(counts_p[iMatter][iPt] * (1. - eff_p[iMatter][iPt])) / ref_p[iMatter][iPt];

      fEfficiency[iMatter]->SetBinContent(iPt + 1, eff_p[iMatter][iPt]);
      fEfficiency[iMatter]->SetBinError(iPt + 1, err_p[iMatter][iPt]);
    }
  }

  for (int iPt = 0; iPt < 40; iPt++) {
    ref_p[2][iPt]    = ref_p[0][iPt] + ref_p[1][iPt];
    counts_p[2][iPt] = counts_p[0][iPt] + counts_p[1][iPt];

    eff_p[2][iPt] = counts_p[2][iPt] / ref_p[2][iPt];
    err_p[2][iPt] = std::sqrt(counts_p[2][iPt] * (1. - eff_p[2][iPt])) / ref_p[2][iPt];

    fEfficiency[2]->SetBinContent(iPt + 1, eff_p[2][iPt]);
    fEfficiency[2]->SetBinError(iPt + 1, err_p[2][iPt]);
  }

  //------------------------------------------------------------
  // efficiency vs ct calculation
  //------------------------------------------------------------

  double counts_c[100];
  double ref_c[100];
  double eff_c[100];
  double err_c[100];

  for (int iCt = 0; iCt < 100; iCt++) {
    ref_c[iCt]    = fHistGenCT->GetBinContent(iCt + 1);
    counts_c[iCt] = fHistRecCT->GetBinContent(iCt + 1);

    eff_c[iCt] = counts_c[iCt] / ref_c[iCt];
    err_c[iCt] = std::sqrt(counts_c[iCt] * (1. - eff_c[iCt])) / ref_c[iCt];

    fEfficiencyCT->SetBinContent(iCt + 1, eff_c[iCt]);
    fEfficiencyCT->SetBinError(iCt + 1, err_c[iCt]);
  }

  //------------------------------------------------------------
  // efficiency vs pT calculation for 4 Centrality classes
  //------------------------------------------------------------

  double counts_cent[4][40];
  double ref_cent[4][40];
  double eff_cent[4][40];
  double err_cent[4][40];

  double num_cent[4];
  double den_cent[4];

  for (int iCent = 0; iCent < 4; iCent++) {
    for (int iPt = 0; iPt < 40; iPt++) {
      ref_cent[iCent][iPt]    = fHistGenCent[iCent]->GetBinContent(iPt + 1);
      counts_cent[iCent][iPt] = fHistRecCent[iCent]->GetBinContent(iPt + 1);

      num_cent[iCent] += counts_cent[iCent][iPt];
      den_cent[iCent] += ref_cent[iCent][iPt];

      eff_cent[iCent][iPt] = counts_cent[iCent][iPt] / ref_cent[iCent][iPt];
      err_cent[iCent][iPt] = std::sqrt(counts_cent[iCent][iPt] * (1. - eff_cent[iCent][iPt])) / ref_cent[iCent][iPt];

      fEfficiencyCent[iCent]->SetBinContent(iPt + 1, eff_cent[iCent][iPt]);
      fEfficiencyCent[iCent]->SetBinError(iPt + 1, err_cent[iCent][iPt]);
    }
  }

  // std::cout << "Eff:  0-10%  --> " << num_cent[0] / den_cent[0] << std::endl;
  // std::cout << "Eff: 10-30%  --> " << num_cent[1] / den_cent[1] << std::endl;
  // std::cout << "Eff: 30-50%  --> " << num_cent[2] / den_cent[2] << std::endl;

  //------------------------------------------------------------
  // efficiency in 10-40 for comparison with Stefano
  //------------------------------------------------------------

  double n_num[10];
  double n_den[10];
  double eff[10];
  double err[10];

  for (int iPt = 0; iPt < 10; iPt++) {
    n_num[iPt] = fHistRecCent[3]->Integral((4 * iPt) + 1, (iPt + 1) * 4);
    n_den[iPt] = fHistGenCent[3]->Integral((4 * iPt) + 1, (iPt + 1) * 4);
    eff[iPt]   = n_num[iPt] / n_den[iPt];
    err[iPt]   = std::sqrt(n_num[iPt] * (1. - eff[iPt])) / n_den[iPt];
    if (iPt < 5) {
      std::cout << "Efficiency " << iPt << "" << iPt + 1 << " --> " << eff[iPt] << " +- " << err[iPt] << std::endl;
      fEfficiencyCentComp->SetBinContent(iPt + 1, eff[iPt]);
      fEfficiencyCentComp->SetBinError(iPt + 1, err[iPt]);
    }
  }
  double num_59 = n_num[5] + n_num[6] + n_num[7] + n_num[8];
  double den_59 = n_den[5] + n_den[6] + n_den[7] + n_den[8];
  double eff_59 = num_59 / den_59;
  double err_59 = std::sqrt(num_59 * (1. - eff_59)) / den_59;
  std::cout << "Efficiency " << 5 << "-" << 9 << " --> " << eff_59 << " +- " << err_59 << std::endl;
  fEfficiencyCentComp->SetBinContent(6, eff_59);
  fEfficiencyCentComp->SetBinError(6, err_59);
  std::cout << std::endl;

  //______________________________________________________________________________

  TFile fOutput("efficiency.root", "RECREATE");

  //------------------------------------------------------------
  // efficiency vs pT Centrality 10-40 plot
  //------------------------------------------------------------

  histo_makeup(fEfficiencyCentComp, kGreenC, "#it{p}_{T} (GeV/#it{c} )", "efficiency #it{x} acceptance");
  fEfficiencyCentComp->GetYaxis()->SetRangeUser(-0.01, 0.41);
  TLegend *le = new TLegend(0.225, 0.626, 0.540, 0.840);
  le->SetFillStyle(0);
  le->SetTextSize(18);
  le->SetHeader("10-40 %", "C");
  le->AddEntry(fEfficiencyCentComp, "{}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H}");
  TCanvas ce("c_10-40", "", 700, 500);
  fEfficiencyCentComp->Draw();
  le->Draw();
  ce.Write();
  ce.Close();

  //------------------------------------------------------------
  // efficiency vs pT plots
  //------------------------------------------------------------
  for (int iMatter = 0; iMatter < 3; iMatter++) {
    histo_makeup(fEfficiency[iMatter], kGreenC, "#it{p}_{T} (GeV/#it{c} )", "efficiency #it{x} acceptance");
    fEfficiency[iMatter]->GetXaxis()->SetRangeUser(0.0, 9.0);
    // fEfficiency[iMatter]->Write();
    TLegend *l = new TLegend(0.225, 0.626, 0.540, 0.840);
    l->SetFillStyle(0);
    l->SetTextSize(18);
    if (iMatter == 2) {
      l->AddEntry(fEfficiency[iMatter], "{}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H}");
    }
    TCanvas c(Form("c_%c", lAM[iMatter]), "", 700, 500);
    fEfficiency[iMatter]->Draw();
    l->Draw();
    c.Write();
    c.Close();
  }

  fEfficiency[0]->SetLineColor(kRedC);

  /// matter-antimatter comparison
  TLegend *l0 = new TLegend(0.225, 0.626, 0.540, 0.840);
  l0->SetFillStyle(0);
  l0->SetTextSize(18);
  l0->AddEntry(fEfficiency[1], "{}^{3}_{#Lambda}H");
  l0->AddEntry(fEfficiency[0], "{}^{3}_{#bar{#Lambda}}#bar{H}");
  ratio_plot(fEfficiency[1], fEfficiency[0], l0, "diff", -0.12, 0.12, "cAM");
  delete l0;

  //------------------------------------------------------------
  // efficiency vs ct plot
  //------------------------------------------------------------
  histo_makeup(fEfficiencyCT, kGreenC, "#it{c}t (cm)", "efficiency #it{x} acceptance");
  fEfficiencyCT->Write();
  TLegend *l = new TLegend(0.225, 0.626, 0.540, 0.840);
  l->SetFillStyle(0);
  l->SetTextSize(18);
  l->AddEntry(fEfficiencyCT, "{}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H}");
  TCanvas c("c_CT", "", 700, 500);
  fEfficiencyCT->Draw();
  l->Draw();
  c.Write();
  c.Close();

  //------------------------------------------------------------
  // efficiency vs pT plots
  //------------------------------------------------------------
  histo_makeup(fEfficiencyCent[0], kRedC, "#it{p}_{T} (GeV/#it{c} )", "efficiency #it{x} acceptance");
  histo_makeup(fEfficiencyCent[1], kBlueC, "#it{p}_{T} (GeV/#it{c} )", "efficiency #it{x} acceptance");
  histo_makeup(fEfficiencyCent[2], kGreenC, "#it{p}_{T} (GeV/#it{c} )", "efficiency #it{x} acceptance");
  fEfficiencyCent[0]->GetXaxis()->SetRangeUser(0.0, 9.0);
  fEfficiencyCent[1]->GetXaxis()->SetRangeUser(0.0, 9.0);
  fEfficiencyCent[2]->GetXaxis()->SetRangeUser(0.0, 9.0);
  fEfficiencyCent[0]->Write();
  fEfficiencyCent[1]->Write();
  fEfficiencyCent[2]->Write();

  TLegend *l_cent = new TLegend(0.225, 0.626, 0.540, 0.840);
  l_cent->SetFillStyle(0);
  l_cent->SetTextSize(18);
  l_cent->SetHeader("   {}^{3}_{#Lambda}H + {}^{3}_{#bar{#Lambda}}#bar{H}");
  l_cent->AddEntry(fEfficiencyCent[0], "0-10 %");
  l_cent->AddEntry(fEfficiencyCent[1], "10-30 %");
  l_cent->AddEntry(fEfficiencyCent[2], "30-50 %");
  TCanvas cCent("c_Cent", "", 700, 500);
  fEfficiencyCent[0]->Draw();
  fEfficiencyCent[1]->Draw("HESAME");
  fEfficiencyCent[2]->Draw("HESAME");
  l_cent->Draw();
  cCent.Write();
  cCent.Close();

  fOutput.Close();

  myFile->Close();
}

//______________________________________________________________________________

//------------------------------------------------------------
// usefull functions
//------------------------------------------------------------

//______________________________________________________________________________

double distance(double *v1, double *v2) {
  double dist = 0;
  for (int i = 0; i < 3; i++) {
    dist += (v1[i] - v2[i]) * (v1[i] - v2[i]);
  }
  return std::sqrt(dist);
}

void histo_makeup(TH1 *h, int color, string xTitle, string yTitle, int lWidth, int msize, int mstyle, string opt,
                  Bool_t stat) {

  h->SetOption(opt.data());
  h->SetStats(stat);
  h->SetMarkerStyle(mstyle);
  h->SetMarkerSize(msize);
  h->SetMarkerColor(color);
  h->SetLineColor(color);
  h->SetLineWidth(lWidth);
  // not controlled by parameters
  // x-axsis settings
  h->GetXaxis()->SetTitle(xTitle.data());
  h->GetXaxis()->SetTitleFont(43);
  h->GetXaxis()->SetTitleSize(24);
  h->GetXaxis()->SetTitleOffset(1.);
  h->GetXaxis()->SetLabelFont(43);
  h->GetXaxis()->SetLabelSize(20);
  // y-axsis settings
  h->GetYaxis()->SetTitle(yTitle.data());
  h->GetYaxis()->SetTitleFont(43);
  h->GetYaxis()->SetTitleSize(24);
  h->GetYaxis()->SetTitleOffset(1.);
  h->GetYaxis()->SetLabelFont(43);
  h->GetYaxis()->SetLabelSize(18);

  h->GetYaxis()->SetRangeUser(-0.02, 0.7);
};

void ratio_plot(TH1 *h1, TH1 *h2, TLegend *l, string mode, float lRange, float uRange, string cName, int cXSize,
                int cYSize, int rColor) {

  TCanvas c(cName.data(), "", cXSize, cYSize);

  TPad *pad1 = new TPad("pad1", "pad1", 0, 0.3, 1, 1.0);
  pad1->SetBottomMargin(0); // Upper and lower plot are joined
  pad1->Draw();             // Draw the upper pad: pad1
  pad1->cd();               // pad1 becomes the current pad

  h1->SetStats(0);
  h1->Draw();

  string h2opt = (string)h2->GetOption() + "SAME";
  h2->Draw(h2opt.data());

  l->Draw();

  TGaxis *axis = new TGaxis(-5, 20, -5, 220, 20, 220, 510, "");
  axis->SetLabelFont(43); // Absolute font size in pixel (precision 3)
  axis->SetLabelSize(15);
  axis->SetTitleFont(43);
  axis->SetTitleSize(20);
  axis->SetTitleOffset(1.3);
  axis->Draw();
  c.Update();

  // lower plot will be in pad
  c.cd(); // Go back to the main canvas before defining pad2
  TPad *pad2 = new TPad("pad2", "pad2", 0, 0.00, 1, 0.3);
  pad2->SetTopMargin(0.0);
  pad2->SetBottomMargin(0.25);
  pad2->SetGridy(1);
  pad2->Draw();
  pad2->cd(); // pad2 becomes the current pad

  // Define the ratio plot
  TH1F *h3 = (TH1F *)h1->Clone("h3");
  h3->SetLineColor(rColor);
  h3->SetMarkerColor(rColor);
  h3->SetMarkerSize(0.5);
  h3->Sumw2();
  h3->SetStats(0); // No statistics on lower plot
  if (mode == "ratio") {
    h3->Divide(h2);
  }
  if (mode == "diff") {
    h3->Add(h2, -1.);
  }
  h3->SetMarkerStyle(21);
  h3->Draw("ep"); // Draw the ratio plot

  // Ratio plot (h3) settings
  h3->SetTitle(""); // Remove the ratio title

  // Y axis ratio plot settings
  string h3Title = mode + "   ";
  h3->GetYaxis()->SetTitle(h3Title.data());
  h3->GetYaxis()->SetNdivisions(505);
  h3->GetYaxis()->SetTitleSize(20);
  h3->GetYaxis()->SetTitleFont(43);
  h3->GetYaxis()->SetTitleOffset(1.3);
  h3->GetYaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
  h3->GetYaxis()->SetLabelSize(13);
  h3->GetYaxis()->SetRangeUser(lRange, uRange);

  // X axis ratio plot settings
  h3->GetXaxis()->SetTitleSize(20);
  h3->GetXaxis()->SetTitleFont(43);
  h3->GetXaxis()->SetTitleOffset(3.);
  h3->GetXaxis()->SetLabelFont(43); // Absolute font size in pixel (precision 3)
  h3->GetXaxis()->SetLabelSize(15);

  c.Write();
  c.Close();

  delete h3;
}
