# this class has been created to generalize the training and to open the file.root just one time
# to achive that alse analysis_utils.py and Significance_Test.py has been modified
import os
import numpy as np
from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import xgboost as xgb
from ROOT import TF1, TFile, gDirectory, TH2D, TH1D
from root_numpy import fill_hist
from sklearn.model_selection import train_test_split
import hyp_analysis_utils as hau
import hyp_plot_utils as hpu



class TrainingAnalysis:

    def __init__(
            self, mode, mc_file_name, bkg_file_name, split):
        self.mode = mode

        print('\n++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT training and testing ... ')
        print('\n++++++++++++++++++++++++++++++++++++++++++++')
        if mode == 3:
            self.df_signal = uproot.open(mc_file_name)[
                'SignalTable'].pandas.df()
            self.df_generated = uproot.open(
                mc_file_name)['GenerateTable'].pandas.df()
            self.df_bkg = uproot.open(bkg_file_name)[
                'BackgroundTable'].pandas.df()

        if mode == 2:
            self.df_signal = uproot.open(mc_file_name)[
                'SignalTable'].pandas.df()
            self.df_generated = uproot.open(
                mc_file_name)['GenTable'].pandas.df()
            self.df_bkg = uproot.open(bkg_file_name)[
                'DataTable'].pandas.df()

        if split == '_antimatter':
            self.df_bkg = self.df_bkg.query('ArmenterosAlpha < 0')
            self.df_signal = self.df_signal.query('ArmenterosAlpha < 0')
            self.df_generated = self.df_generated.query('matter < 0.5')
        if split == '_matter':
            self.df_bkg = self.df_bkg.query('ArmenterosAlpha > 0')
            self.df_signal = self.df_signal.query('ArmenterosAlpha > 0')
            self.df_generated = self.df_generated.query('matter > 0.5')

        self.df_signal['y'] = 1
        self.df_bkg['y'] = 0

    def compute_preselection_efficiency(self, cent_class, ct_bins, pt_bins, split_type):
        cent_cut = '{}<centrality<{}'.format(cent_class[0], cent_class[1])
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_{}'.format(
            self.mode)]
        filename = efficiencies_path + \
            "/PreselEff_cen_{}{}{}.root".format(
                cent_class[0], cent_class[1], split_type)
        t_file = TFile(filename, "recreate")
        pres_histo = TH2D("PreselEff", ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm);Preselection efficiency',
                          len(pt_bins)-1, np.array(pt_bins, 'double'), len(ct_bins)-1, np.array(ct_bins, 'double'))
        gen_histo = TH2D("bkg_histo", ';#it{p}_{T} (GeV/#it{c});c#it{t} (cm); bkg_candidates',
                         len(pt_bins)-1, np.array(pt_bins, 'double'), len(ct_bins)-1, np.array(ct_bins, 'double'))

        fill_hist(pres_histo, self.df_signal.query(
            cent_cut)[['HypCandPt', 'ct']])
        fill_hist(gen_histo, self.df_generated.query(cent_cut)[['pT', 'ct']])
        pres_histo.Divide(gen_histo)
        pres_histo.Write()
        t_file.Close()
        return pres_histo

    def prepare_dataframe(self, training_columns, cent_class, pt_range,
                          ct_range, bkg_reduct=False, bkg_factor=1):
        data_range = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1])

        if bkg_reduct:
            n_bkg = int(len(bkg) * bkg_factor)
            if n_bkg < len(bkg):
                bkg = bkg.sample(n=n_bkg)
        sig = self.df_signal.query(data_range)
        bkg = self.df_bkg.query(data_range)
        print('\nNumber of signal candidates: {}'.format(len(sig)))
        print('Number of background candidates: {}\n'.format(len(bkg)))
        df = pd.concat([self.df_signal.query(data_range),
                        self.df_bkg.query(data_range)])
        train_set, test_set, y_train, y_test = train_test_split(
            df[training_columns+['InvMass']], df['y'], test_size=0.5, random_state=42)
        return [train_set, y_train, test_set, y_test]

    def compute_BDT_efficiency(self, y_test, y_pred, ext_params_efficiency):
        efficiency, threshold = analysis_utils.bdt_efficiency_array(
            y_test, y_pred, n_points=1000)
        index_list = []
        for eff_par in ext_params_efficiency:
            index_list.append(np.argmin(np.abs(efficiency-eff_par)))
        eff_score_array = np.vstack(
            (np.round(efficiency[index_list], 2), threshold[index_list]))
        return eff_score_array

    def compute_and_save_MC_sigma_array(self, data, eff_score_array, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):
        inv_mass_array = np.array(np.arange(2.97, 3.04225, 0.00225))
        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split_string)
        sigma_path = os.environ['HYPERML_UTILS_{}'.format(
            self.mode)]+'/FixedSigma'
        if not os.path.exists(sigma_path):
            os.makedirs(sigma_path)
        filename_sigma = sigma_path + "/sigma_array" + info_string + '.npy'
        sigma_list = []
        data[2]['Score'] = data[2]['Score'].astype(float)
        for score in eff_score_array[1]:
            counts = data[2][data[3].astype(bool)].query("Score>@score")['InvMass']
            histo_inv = TH1D("InvMassHist", '; (GeV/#it{c^2}) ; Raw counts', len(inv_mass_array)-1, inv_mass_array)
            fill_hist(histo_inv, counts)
            histo_inv.Fit('gaus', 'Q')
            # if (pt_range[0] ==2 and split_string == '_matter' and 1<score<2):
            #     tfile = TFile('histo_sigma.root', "recreate")
            #     histo_inv.Write()
            #     tfile.Close()
            sigma = histo_inv.GetFunction('gaus').GetParameter(2)
            del histo_inv
            sigma_list.append(sigma)
        np.save(filename_sigma, np.array(sigma_list))


    def save_ML_analysis(self, model_handler, efficiency_score_array, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):

        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split_string)
        models_path = os.environ['HYPERML_MODELS_{}'.format(
            self.mode)]+'/models'
        handlers_path = os.environ['HYPERML_MODELS_{}'.format(
            self.mode)]+'/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_{}'.format(
            self.mode)]

        if not os.path.exists(models_path):
            os.makedirs(models_path)
        if not os.path.exists(handlers_path):
            os.makedirs(handlers_path)

        filename_handler = handlers_path + '/model_handler'+info_string+'.pkl'
        filename_model = models_path + '/BDT'+info_string+'.model'
        filename_efficiencies = efficiencies_path + '/Eff_Score'+info_string+'.npy'

        model_handler.dump_model_handler(filename_handler)
        model_handler.dump_original_model(filename_model, xgb_format=True)
        np.save(filename_efficiencies, efficiency_score_array)

        print('ML analysis results saved.\n')

    def save_ML_plots(self, model_handler, data, efficiency_score_array, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):
        fig_path = os.environ['HYPERML_FIGURES_{}'.format(self.mode)]
        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split_string)

        bdt_eff_path = fig_path + '/Efficiency'
        feat_imp_path = fig_path + '/FeatureImportance'

        BDT_EFFICIENCY_PLOT = plot_utils.plot_bdt_eff(
            efficiency_score_array[1], efficiency_score_array[0])
        if not os.path.exists(bdt_eff_path):
            os.makedirs(bdt_eff_path)
        plt.savefig(bdt_eff_path + '/BDT_Eff' + info_string + '.pdf')
        FEATURES_IMPORTANCE = plot_utils.plot_feature_imp(
            data[2], data[3], model_handler)
        if not os.path.exists(feat_imp_path):
            os.makedirs(feat_imp_path)
        plt.savefig(feat_imp_path + '/Feat_Imp' + info_string + '.pdf')

        print('ML plots saved.\n')


class ModelApplication:

    def __init__(self, mode, data_path, cent_classes, split):

        print('\n++++++++++++++++++++++++++++++++++++++++++++')
        print('\nStarting BDT appplication and signal extraction ... ')

        self.mode = mode
        self.df_data = uproot.open(data_path)['DataTable'].pandas.df()
        if mode == 2:
            self.hist_centrality = uproot.open(data_path)['EventCounter']
            self.n_events = []
            for cent in cent_classes:
                self.n_events.append(
                    sum(self.hist_centrality[cent[0]+1:cent[1]]))
        print('\nNumber of events: ', int(sum(self.hist_centrality[:])))
        if split == '_antimatter':
            self.df_data = self.df_data.query('ArmenterosAlpha < 0')
            print('\nNumber of anti-hyper-candidates: {}'.format(len(self.df_data)))
        if split == '_matter':
            self.df_data = self.df_data.query('ArmenterosAlpha > 0')
            print('Number of hyper-candidates: {}'.format(len(self.df_data)))
        print('\n++++++++++++++++++++++++++++++++++++++++++++')



    def load_preselection_efficiency(self, cent_class, split_type):
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_{}'.format(
            self.mode)]
        filename_efficiencies = efficiencies_path + "/PreselEff_cen_{}{}{}.root".format(cent_class[0],
                                                                                        cent_class[1], split_type)

        tfile =  TFile(filename_efficiencies)
        self.presel_histo = tfile.Get("PreselEff")
        self.presel_histo.SetDirectory(0)


    def load_ML_analysis(self, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):

        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split_string)

        handlers_path = os.environ['HYPERML_MODELS_{}'.format(
            self.mode)]+'/handlers'
        efficiencies_path = os.environ['HYPERML_EFFICIENCIES_{}'.format(
            self.mode)]

        filename_handler = handlers_path + '/model_handler'+info_string+'.pkl'
        filename_efficiencies = efficiencies_path + '/Eff_Score'+info_string+'.npy'

        efficiency_score_array = np.load(filename_efficiencies)
        model_handler = ModelHandler()
        model_handler.load_model_handler(filename_handler)
        return efficiency_score_array, model_handler

    def return_preselection_efficiency(self, ptbin_index, ctbin_index):
        return self.presel_histo.GetBinContent(ptbin_index, ctbin_index)


    def load_sigma_array(self, cent_class=[0, 90], pt_range=[0, 10], ct_range=[0, 100], split_string=''):

        info_string = '_{}{}_{}{}_{}{}{}'.format(cent_class[0], cent_class[1], pt_range[0],
                                                 pt_range[1], ct_range[0], ct_range[1], split_string)
        sigma_path = os.environ['HYPERML_UTILS_{}'.format(
            self.mode)]+'/FixedSigma'
        filename_sigma = sigma_path + "/sigma_array" + info_string + '.npy'
        return np.load(filename_sigma)



    def apply_BDT_to_data(self, model_handler, cent_class, pt_bins, ct_bins, training_columns):
        print('\nApplying BDT to data: ...')
        data_range = '{}<ct<{} and {}<HypCandPt<{} and {}<centrality<{}'.format(
            ct_bins[0], ct_bins[1], pt_bins[0], pt_bins[1], cent_class[0], cent_class[1])
        df_cut = self.df_data.query(data_range)
        df_cut.insert(0, 'Score', model_handler.predict(df_cut[training_columns]))
        df_cut = df_cut.loc[:,['Score', 'ct', 'HypCandPt', 'InvMass']]
        print('Application: Done!')
        return df_cut

    def significance_scan(self, df_bkg ,pre_selection_efficiency, efficiency_score_array,  cent_class=[0, 100],
                          ct_range=[0, 100], pt_range=[0, 10], split_string=''):

        print('\nSignificance scan: ...')

        bdt_efficiency = efficiency_score_array[0]
        threshold_space = efficiency_score_array[1]

        expected_signal = []
        significance = []
        significance_error = []
        significance_custom = []
        significance_custom_error = []

        hyp_lifetime = 206

        bw_file = TFile(os.environ['HYPERML_UTILS'] + '/BlastWaveFits.root')
        bw = [bw_file.Get("BlastWave/BlastWave{}".format(i))
              for i in [0, 1, 2]]

        for index, tsd in enumerate(threshold_space):
            df_selected = df_bkg.query('Score>@tsd')

            counts, bins = np.histogram(
                df_selected['InvMass'], bins=45, range=[2.96, 3.05])
            bin_centers = 0.5 * (bins[1:] + bins[:-1])

            side_map = (bin_centers < 2.98) + (bin_centers > 3.005)
            mass_map = np.logical_not(side_map)
            bins_side = bin_centers[side_map]
            counts_side = counts[side_map]

            h, residuals, _, _, _ = np.polyfit(
                bins_side, counts_side, 2, full=True)
            y = np.polyval(h, bins_side)

            exp_signal_ctint = hau.expected_signal_counts(
                bw, pt_range, pre_selection_efficiency * bdt_efficiency[index],
                cent_class, self.hist_centrality)

            if split_string is not '':
                exp_signal_ctint = 0.5 * exp_signal_ctint

            ctrange_correction = hau.expo(
                ct_range[0], hyp_lifetime)-hau.expo(ct_range[1], hyp_lifetime)

            exp_signal = exp_signal_ctint * ctrange_correction
            exp_background = sum(np.polyval(h, bin_centers[mass_map]))

            expected_signal.append(exp_signal)

            if (exp_background < 0):
                exp_background = 0

            sig = exp_signal / np.sqrt(exp_signal + exp_background + 1e-10)
            sig_error = hau.significance_error(exp_signal, exp_background)

            significance.append(sig)
            significance_error.append(sig_error)

            sig_custom = sig * bdt_efficiency[index]
            sig_custom_error = sig_error * bdt_efficiency[index]

            significance_custom.append(sig_custom)
            significance_custom_error.append(sig_custom_error)

        nevents = sum(self.hist_centrality[cent_class[0]+1:cent_class[1]])

        max_index = np.argmax(significance_custom)
        max_score = threshold_space[max_index]
        max_significance = significance_custom[max_index]
        data_range_array = [ct_range[0], ct_range[1], pt_range[0], pt_range[1], cent_class[0], cent_class[1]]
        hpu.plot_significance_scan(
            max_index, significance_custom, significance_custom_error, expected_signal, df_bkg,
            threshold_space, data_range_array, bin_centers, nevents, self.mode, split_string=split_string)

        bdt_eff_max_score = bdt_efficiency[max_index]

        print('Significance scan: Done!')

        # return max_score, bdt_eff_max_score, max_significance
        return bdt_eff_max_score, max_score
