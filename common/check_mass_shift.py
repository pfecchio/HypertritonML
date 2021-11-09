import ROOT

ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR);
ROOT.RooMsgService.instance().setSilentMode(True)
ROOT.gErrorIgnoreLevel=ROOT.kError

### MC gen ----> MC rec
### Hyp mass ----> Data rec


MC_gen_mass = 2.991  #####massa MC generato
hyp_mass = 3. #####massa vera dati
reco_bias = 0.002 ##### bias dovuto alla ricostruzione


mass = ROOT.RooRealVar('m', 'm_{^{3}He+#pi}', 2., 4., 'GeV/c^{2}') 



#### genero due distribuzioni di segnale, centrate su massa MC + reco_shift , estraggo per una la KDE e poi verifico sulla seconda il bias dovuto al fit con KDE
mu = ROOT.RooRealVar('mu', 'hypertriton mass', MC_gen_mass + reco_bias, 2.989, 2.993, 'GeV/c^{2}')
sigma = ROOT.RooRealVar('sigma', 'hypertriton width', 0.002, 0.004, 'GeV/c^{2}')
gauss_signal_pdf = ROOT.RooGaussian("gauss","gaussian PDF", mass, mu, sigma)
signal_dataset1 = gauss_signal_pdf.generate(mass, 1e4)
signal_dataset2 = gauss_signal_pdf.generate(mass, 1e5)



#### genero distribuzione di dati, centrata su massa dell'ipertrizio + reco shift
mu_data = ROOT.RooRealVar('mu data', 'hypertriton mass data', hyp_mass + reco_bias, 2., 4., 'GeV/c^{2}')
sigma_data = ROOT.RooRealVar('sigma data', 'hypertriton width data', 0.002, 0.0001, 0.004, 'GeV/c^{2}')
gauss_data_pdf = ROOT.RooGaussian("gauss data","gaussian PDF", mass, mu_data, sigma_data)
data_dataset = gauss_data_pdf.generate(mass, 1e5)


### costruisco KDE e valuto bias dovuto al fit
delta_mass = ROOT.RooRealVar('delta_m', '#Delta m', -1., 1., 'GeV/c^{2}')
shift_mass = ROOT.RooAddition('shift_m', "m + #Delta m", ROOT.RooArgList(mass, delta_mass))
signal_kde = ROOT.RooKeysPdf('signal kde', 'signal kde', shift_mass, mass, signal_dataset1, ROOT.RooKeysPdf.NoMirror, 2.)
fit_results_mc = signal_kde.fitTo(signal_dataset2, ROOT.RooFit.Range(2, 4), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())
fit_bias = delta_mass.getVal()
print("Fit bias: ", fit_bias)


### fitto sui dati, e valuto il segno del bias
fit_results_data = signal_kde.fitTo(data_dataset, ROOT.RooFit.Range(2., 4.), ROOT.RooFit.NumCPU(32), ROOT.RooFit.Save())
print("Shift from data: ", delta_mass.getVal())

print("Corrected mass without fit bias: ", MC_gen_mass - delta_mass.getVal(), ", Relative error: ",  (hyp_mass - (MC_gen_mass - delta_mass.getVal()))/hyp_mass)
print("Corrected mass - fit bias: ", MC_gen_mass - delta_mass.getVal() - fit_bias, ", Relative error: ",  (hyp_mass - (MC_gen_mass - delta_mass.getVal() - fit_bias))/hyp_mass)
print("Corrected mass + fit bias: ", MC_gen_mass - delta_mass.getVal() + fit_bias,  ", Relative error: ",  (hyp_mass - (MC_gen_mass - delta_mass.getVal() + fit_bias))/hyp_mass)



