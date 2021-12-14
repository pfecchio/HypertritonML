#include "Riostream.h"
#include <TBox.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TH2.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLegendEntry.h>
#include <TLine.h>
#include <TPad.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TColor.h>


void CollectionPlotMass() {
  gStyle->SetOptStat(0);

  const Int_t kMarkTyp   = 20;    // marker type
  const Int_t kMarkCol   = 1;     //...and color
  const Int_t kFont      = 42;
  const Int_t kLineWidth = 2;

//   Float_t bind[N]       = {0.41, 0.08, -0.24, 0.27, 0.4};
//   Float_t err_y_low[N]  = {0.12, 0.07, 0.22, 0.08, 0.12};
//   Float_t err_y_high[N] = {0.12, 0.07, 0.22, 0.08, 0.12};

  const int N           = 5;
  Float_t point[N]      = {1, 2, 3, 4, 5};
  Float_t bind[N]       = {0.41, 0.08, -0.24, 0.27, 0.4};
  Float_t err_x[N]      = {0, 0, 0, 0, 0};
  Float_t err_y_low[N]  = {0.12, 0.07, 0.22, 0.08, 0.12};
  Float_t err_y_high[N] = {0.12, 0.07, 0.22, 0.08, 0.12};
  Float_t errsyst_y[N]  = {0., 0., 0., 0., 0.11};
  Float_t errsyst_x[N]  = {0, 0, 0, 0, 0.1};

  Float_t point_a[1]      = {6};
  Float_t bind_a[1]       = {0.077};
  Float_t err_x_a[1]      = {0};
  Float_t err_y_low_a[1]  = {0.063};
  Float_t err_y_high_a[1] = {0.063};
  Float_t errsyst_y_a[1]  = {0.030};
  Float_t errsyst_x_a[1]  = {0.1};

  TGraphAsymmErrors* gSpect;
  gSpect = new TGraphAsymmErrors(N, point, bind, err_x, err_x, err_y_low, err_y_high);

  TGraphAsymmErrors* gSpect2;    // syst. err.
  gSpect2 = new TGraphAsymmErrors(N, point, bind, errsyst_x, errsyst_x, errsyst_y, errsyst_y);

  TGraphAsymmErrors* gSpect_alice =
      new TGraphAsymmErrors(1, point_a, bind_a, err_x_a, err_x_a, err_y_low_a, err_y_high_a);
  TGraphAsymmErrors* gSpect2_alice =
      new TGraphAsymmErrors(1, point_a, bind_a, errsyst_x_a, errsyst_x_a, errsyst_y_a, errsyst_y_a);

  TLine* theo1 = new TLine(0., 0.10, 7., 0.10);
  theo1->SetLineWidth(kLineWidth);
  theo1->SetLineStyle(2);
  theo1->SetLineColor(kPurpleC);

  TLine* theo2 = new TLine(0., 0.262, 7., 0.262);
  theo2->SetLineWidth(kLineWidth);
  theo2->SetLineStyle(9);
  theo2->SetLineColor(kOrangeC);

  TLine* theo3 = new TLine(0., 0.23, 7., 0.23);
  theo3->SetLineWidth(kLineWidth);
  theo3->SetLineStyle(10);
  theo3->SetLineColor(kMagentaC);

  const int kBCT = TColor::GetColor("#b8d4ff");

  TBox* boxtheo = new TBox(0., 0.046, 7., 0.135);
  boxtheo->SetFillColor(kBCT);
  boxtheo->SetFillStyle(1001);
  boxtheo->SetLineColor(kWhite);
  boxtheo->SetLineWidth(0);

  TCanvas* c1 = new TCanvas("B_lambda_collection", "B_lambda_collection", 550, 525);
  c1->SetTopMargin(0.075);
  c1->SetBottomMargin(0.075);
  c1->SetLeftMargin(0.125);
  c1->SetRightMargin(0.1);
  c1->cd(1); 

  TH2D* ho1 = new TH2D("ho1", "ho1", 1000, 0., 7, 1000, -5, 5);    //...just for frame
  ho1->GetYaxis()->SetTitle("#it{B}_{#Lambda} (MeV)");
  ho1->GetXaxis()->SetTitleOffset(1.1);
  ho1->GetYaxis()->SetTitleOffset(1.1);
  ho1->GetYaxis()->SetLabelOffset(.01);
  ho1->GetYaxis()->SetRangeUser(-0.58, 1.28);
  ho1->GetXaxis()->SetLabelOffset(999);
  ho1->GetXaxis()->SetLabelSize(0);
  ho1->GetXaxis()->SetTickLength(0.);
  ho1->SetTitleSize(0.05, "XY");
  ho1->SetTitleFont(kFont, "XY");
  ho1->SetLabelSize(0.045, "XY");
  ho1->SetLabelFont(kFont, "XY");
  ho1->SetMarkerStyle(kFullCircle);
  ho1->SetTitle("");
  ho1->Draw("");

  gSpect->SetMarkerStyle(kMarkTyp);
  gSpect->SetMarkerSize(1.2);
  gSpect->SetMarkerColor(kMarkCol);
  gSpect->SetLineColor(kMarkCol);
  gSpect->SetLineWidth(1);

  gSpect2->SetMarkerStyle(0);
  gSpect2->SetMarkerColor(kMarkCol);
  gSpect2->SetMarkerSize(0.1);
  gSpect2->SetLineStyle(1);
  gSpect2->SetLineColor(kMarkCol);
  gSpect2->SetLineWidth(1);
  gSpect2->SetFillColor(15);
  gSpect2->SetFillStyle(1001);

  gSpect_alice->SetMarkerStyle(kFullDiamond);
  gSpect_alice->SetMarkerSize(2.);
  gSpect_alice->SetMarkerColor(kRedC);
  gSpect_alice->SetLineColor(kRedC);
  gSpect_alice->SetLineWidth(1);

  gSpect2_alice->SetMarkerStyle(0);
  gSpect2_alice->SetMarkerColor(kRedCT);
  gSpect2_alice->SetMarkerSize(0.1);
  gSpect2_alice->SetLineStyle(1);
  gSpect2_alice->SetLineColor(kRedCT);
  gSpect2_alice->SetLineWidth(1);
  gSpect2_alice->SetFillColor(kRedCT);
  gSpect2_alice->SetFillStyle(1001);

  boxtheo->Draw("SAME");
  //   Fcn->Draw("SAME");
  theo1->Draw("SAME");
  theo2->Draw("SAME");
  theo3->Draw("SAME");
  //   theo4->Draw("SAME");
  gSpect2->Draw("spe2");
  gSpect->Draw("pzsame");
  gSpect2_alice->Draw("spe2");
  gSpect_alice->Draw("pzsame");

  //   TLegend* leg1 = new TLegend(0.160, 0.755, 0.684, 0.905); // for 2 legends
  TLegend* leg1 = new TLegend(0.170, 0.758, 0.806, 0.908);
  leg1->SetNColumns(2);
  leg1->SetFillStyle(0);
  leg1->SetMargin(0.2);    // separation symbol-text
  leg1->SetBorderSize(0);
  leg1->SetTextFont(kFont);
  leg1->SetTextSizePixels(20);
  leg1->SetEntrySeparation(0.1);
  leg1->SetHeader("Theoretical calculations");
  leg1->AddEntry(theo1, "NPB47 (1972) 109-137", "fl");
  leg1->AddEntry(theo2, "PRC77 (2008) 027001  ", "fl");
  leg1->AddEntry(theo3, "arXiv:1711.07521", "fl");
  leg1->AddEntry(boxtheo, "EPJA(2020) 56", "f");
  leg1->Draw();

  TLegend* leg2 = new TLegend(0.164, 0.638, 0.651, 0.746);
  leg2->SetNColumns(2);
  leg2->SetFillStyle(0);
  leg2->SetMargin(0.15);    // separation symbol-text
  leg2->SetBorderSize(0);
  leg2->SetTextFont(kFont);
  leg2->SetTextSizePixels(20);
  leg2->SetEntrySeparation(0.1);
  leg2->AddEntry(gSpect, "Erlier measurements", "pe");
  //   leg2->AddEntry(Fcn, "B_{#Lambda} = 0", "L");
  TLegendEntry* e1 = leg2->AddEntry(gSpect_alice, "This work", "pe");
  e1->SetTextColor(kRedC);
  //   leg2->Draw();

  //   TLatex* note = new TLatex();
  //   note->SetNDC(kTRUE);
  //   note->SetTextColor(1);
  //   note->SetTextFont(42);
  //   note->SetTextSize(16);
  //   note->DrawLatex(0.175, 0.84,
  //                   Form("#splitline{Erlier measurements recalibrated with the}{latest #pi, p, d,^{3}He and #Lambda
  //                   mass "
  //                        "measurements}"));

  TLatex* lat1 = new TLatex();
  lat1->SetNDC(kTRUE);
  lat1->SetTextColor(1);
  lat1->SetTextFont(kFont);
  lat1->SetTextSize(0.04);
  lat1->DrawLatex(0.170, 0.600, Form("NPB1 (1967) 105"));

  TLatex* lat2 = new TLatex();
  lat2->SetNDC(kTRUE);
  lat2->SetTextColor(1);
  lat2->SetTextFont(kFont);
  lat2->SetTextSize(0.04);
  lat2->DrawLatex(0.2, 0.30, Form("NPB4 (1968) 511"));

  TLatex* lat3 = new TLatex();
  lat3->SetNDC(kTRUE);
  lat3->SetTextColor(1);
  lat3->SetTextFont(kFont);
  lat3->SetTextSize(0.04);
  lat3->DrawLatex(0.47, 0.12, Form("PRD1 (1970) 66"));

  TLatex* lat4 = new TLatex();
  lat4->SetNDC(kTRUE);
  lat4->SetTextColor(1);
  lat4->SetTextFont(kFont);
  lat4->SetTextSize(0.04);
  lat4->DrawLatex(0.43, 0.515, Form("NPB52 (1973) 1"));

  TLatex* lat5 = new TLatex();
  lat5->SetNDC(kTRUE);
  lat5->SetTextColor(1);
  lat5->SetTextFont(kFont);
  lat5->SetTextSize(0.04);
  lat5->DrawLatex(0.600, 0.600, Form("Nat. Phys 16 (2020)  "));

  TLatex* lat6 = new TLatex();
  lat6->SetNDC(kTRUE);
  lat6->SetTextColor(kRedC);
  lat6->SetTextFont(kFont);
  lat6->SetTextSizePixels(22);
  lat6->DrawLatex(0.68, 0.25, "ALICE");

  //   TPaveText* pavunc = new TPaveText(0.7, 0.12, 0.85, 0.15, "blNDC");
  //   pavunc->SetTextFont(43);
  //   pavunc->SetTextSize(20);
  //   pavunc->SetBorderSize(0);
  //   pavunc->SetFillStyle(0);
  //   pavunc->AddText("Uncertainties: stat. (bars), syst. (boxes)");
  //   pavunc->Draw();

  gPad->RedrawAxis();

  c1->SaveAs("blambda_collection.eps");
  c1->SaveAs("blambda_collection.png");

}
