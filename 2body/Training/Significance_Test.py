from scipy.optimize import curve_fit
import numpy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle
from scipy.stats import norm
from scipy import stats

import os
from ROOT import TF1,TFile,gDirectory,vector




def Yield(pt_bin,Centrality_bin):
    bwFile=TFile("../fitsM.root")
    Scale_Factor = [3.37e-5,1.28e-5,0.77e-5,0.183e-5]

    if Centrality_bin==[30,50]:
        Bl1=bwFile.Get("BlastWave/BlastWave1")
        Bl2=bwFile.Get("BlastWave/BlastWave2")
        index_cen=2
        pT_yield=2*Scale_Factor[index_cen]*(Bl1.Integral(pt_bin[0],pt_bin[1],1e-8)+Bl2.Integral(pt_bin[0],pt_bin[1],1e-8))/(pt_bin[1]-pt_bin[0])/(Bl1.Integral(0,10,1e-8)+Bl2.Integral(0,10,1e-8))
    else: 
        if Centrality_bin==[0,10]:
            BlastWave=bwFile.Get("BlastWave/BlastWave0")
            index_cen=0   
        elif Centrality_bin==[10,30]:
            BlastWave=bwFile.Get("BlastWave/BlastWave1")
            index_cen=1

        elif Centrality_bin==[50,90]:
            BlastWave=bwFile.Get("BlastWave/BlastWave2")   
            index_cen=3
        else:
            return "Centrality class not allowed"
              
        Integral = BlastWave.Integral(0,10,1e-8)
        #last values obtained by a exponential fit
        pT_yield=2*Scale_Factor[index_cen]*BlastWave.Integral(pt_bin[0],pt_bin[1],1e-8)/(pt_bin[1]-pt_bin[0])/Integral 
         
    #BlastWave.SetNormalized(1)
    return pT_yield


def SignificanceError(sig,bkg,pT_bin,Centrality_bin):
    yield_meas = Yield(pT_bin,Centrality_bin)
    err_sig=np.sqrt(yield_meas)/(yield_meas+1e-10)*sig
    err_bkg=np.sqrt(bkg)
    err_sig=np.sqrt(sig)
    err_1=(np.sqrt(sig+bkg+1e-10)-sig*(1/(2*np.sqrt(sig+bkg+1e-10))))/(sig+bkg+1e-10)
    err_2=sig/(2*(sig+bkg+1e-10)**(3/2))
    return abs(err_1)*err_sig+abs(err_2)*err_bkg


def ExpectedSignal(eff_bdt, pT_bin,n_ev,eff_V0,Centrality_bin):
    yield_meas = Yield(pT_bin,Centrality_bin) 
    return int(round(n_ev*yield_meas*(pT_bin[1]-pT_bin[0])*eff_V0*eff_bdt))


def expo(x,tau):
    return np.exp(-x/tau/0.029979245800)

def SignificanceScan(df,ct_cut,pt_cut,centrality_cut,efficiency_array,eff_pres,n_ev,custom=False,draw=False):    
  
    ct_min = ct_cut[0]
    ct_max = ct_cut[1]
    pt_max = pt_cut[1]
    pt_min = pt_cut[0]
    centrality_max = centrality_cut[1]
    centrality_min = centrality_cut[0]
    signal_array = []
    significance_array = []
    custom_significance_array = []
    error_array=[]
    score_list = np.linspace(-3,12.5,156)
    index = 0
    HyTrLifetime = 206
    for i in score_list:
        df_score = df.query('Score>@i and @ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max')
        counts,bins = np.histogram(df_score['InvMass'],bins=30,range=[2.96,3.05])
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        sidemap = (bin_centers<2.9923-3*0.0025) + (bin_centers>2.9923+3*0.0025)
        massmap = np.logical_not(sidemap)
        bins_side = bin_centers[sidemap]
        counts_side = counts[sidemap]
        h, residuals, _, _, _ = np.polyfit(bins_side,counts_side,2,full=True)
        y = np.polyval(h,bins_side)
        
        YpTt = ExpectedSignal(efficiency_array[index],pt_cut,n_ev,eff_pres,centrality_cut)
        Yct = -(expo(ct_max,HyTrLifetime)-expo(ct_min,HyTrLifetime))/(ct_max-ct_min)*HyTrLifetime*0.029979245800
        signal=YpTt*Yct
        
        bkg = sum(np.polyval(h,bin_centers[massmap]))
        significance = signal/np.sqrt(signal+bkg+1e-10)
        signal_array.append(signal)
        error_array.append(SignificanceError(signal,bkg,pt_cut,centrality_cut))
        significance_array.append(significance)
        custom_significance = significance*efficiency_array[index]
        custom_significance_array.append(custom_significance)
        index += 1
    significance_array=np.asarray(significance_array)
    error_array=np.asarray(error_array)
    
    if custom==True:
        max_index = np.argmax(custom_significance_array)
    else:
        max_index = np.argmax(significance_array)
        
    max_score = score_list[max_index]
    sign = significance_array[max_index]
    custom_sign = custom_significance_array[max_index]
    ryield = signal_array[max_index]
    df_cut = df.query('Score>@max_score and @ct_min<Ct<@ct_max and @pt_min<V0pt<@pt_max and @centrality_min<Centrality<@centrality_max')
    counts_mc_0 = norm.pdf(bin_centers,loc=2.992,scale=0.0025)
    counts_mc = (ryield/sum(counts_mc_0))*counts_mc_0
    counts_data,_ = np.histogram(df_cut['InvMass'],bins=30,range=[2.96,3.05])
    h = np.polyfit(bins_side,counts_data[sidemap],2)
    counts_bkg = np.polyval(h,bin_centers)
    counts_tot = counts_bkg+counts_mc
    fig, axs = plt.subplots(1,2, figsize=(12, 4)) 
    axs[0].set_xlabel('Score')
    axs[0].tick_params(axis="x", direction="in")
    axs[0].tick_params(axis="y", direction="in")
    
    
    if custom==True:
        axs[0].set_ylabel('Significance x Efficiency')
        axs[0].plot(score_list,custom_significance_array,'b',label='Expected significance')
        a=custom_significance_array-error_array*efficiency_array
        b=custom_significance_array+error_array*efficiency_array
        axs[0].fill_between(score_list,a,b,facecolor='deepskyblue',label=r'$ \pm 1\sigma$')
        axs[0].grid()
        
    else:
        axs[0].set_ylabel('Significance')
        axs[0].plot(score_list,significance_array,'b',label='Expected significance')
        a=significance_array-error_array
        b=significance_array+error_array
        axs[0].fill_between(score_list,a,b,facecolor='deepskyblue',label=r'$ \pm 1\sigma$')
        axs[0].grid()

    axs[0].legend(loc='upper left')
    plt.suptitle(r"%1.f $ \leq \rm{p}_{T} \leq $ %1.f, Cut Score = %0.2f, Significance/Events = %0.4f$x10^{-4}$, Significance x Efficiency = %0.2f , Raw yield = %0.2f" %(pt_min,pt_max,max_score,(sign/np.sqrt(n_ev))*1e4,custom_sign,ryield))
    
    yerr_data = np.sqrt(counts_data[sidemap])
    yerr_tot = np.sqrt(counts_tot[massmap])
    
    axs[1].errorbar(bin_centers[sidemap],counts_data[sidemap],yerr=yerr_data,fmt='.',ecolor='k',color='b',elinewidth=1.,label='Data')
    axs[1].errorbar(bin_centers[massmap],counts_tot[massmap],yerr=yerr_tot,fmt='.',ecolor='k',color='r',elinewidth=1.,label='Pseudodata')    
    axs[1].plot(bin_centers[sidemap],counts_bkg[sidemap],'g-',label='Background fit')
    x=np.linspace(2.9923-3*0.0025,2.9923+3*0.0025,1000)
    gaussian_counts=norm.pdf(x,loc=2.992,scale=0.0025)
    gaussian_counts=(ryield/sum(counts_mc_0))*gaussian_counts+np.polyval(h,x)
    axs[1].plot(x,gaussian_counts,'y',color='orange',label='Gaussian model')
    axs[1].set_xlabel(r"$m_{\ ^{3}He+\pi^{-}}$")
    axs[1].set_ylabel(r"Events /  $3.6\ \rm{MeV}/c^{2}$") 
    axs[1].tick_params(axis="x", direction="in")
    axs[1].tick_params(axis="y", direction="in")    
    axs[1].legend(loc=(0.37,0.47))
    plt.ylim(bottom=0)
    textstr = '\n'.join((
    r"%1.f GeV/c $ \leq \rm{p}_{T} < $ %1.f GeV/c " %(pt_min,pt_max,),
    r' Significance/Sqrt(Events) = %0.4f$x10^{-4}$' % ((sign/np.sqrt(n_ev))*1e4, )))
    props = dict(boxstyle='round',facecolor='white', alpha=0,)
    axs[1].text(0.37, 0.95, textstr, transform=axs[1].transAxes,
        verticalalignment='top', bbox=props)
    if draw is True:
        plt.show()

    filename = 'Significance_Ct_{:.2f}_{:.2f}_pT_{:.2f}_{:.2f}_Cen_{:.2f}_{:.2f}.pdf'.format(ct_cut[0],ct_cut[1],pt_cut[0],pt_cut[1],centrality_cut[0],centrality_cut[1])

    if not os.path.exists(os.environ['HYPERML_FIGURES']+'/Significance/'):
        os.makedirs(os.environ['HYPERML_FIGURES']+'/Significance/')
    plt.savefig(os.environ['HYPERML_FIGURES']+'/Significance/'+filename)
    plt.close()
    return max_score

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def TestOnData(df,score,pt,n_ev):
    df_score = df.query('Score>@score and V0pt>=@pt[0] and V0pt<=@pt[1]')
    counts,bins = np.histogram(df_score['InvMass'],bins=30,range=[2.96,3.05])
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    sidemap = (bin_centers<2.9923-3*0.0025) + (bin_centers>2.9923+3*0.0025)
    massmap = np.logical_not(sidemap)
    bins_side = bin_centers[sidemap]
    counts_side = counts[sidemap]
    h = np.polyfit(bins_side,counts_side,2)
    y = np.polyval(h,bins_side)
    counts_bkg = np.polyval(h,bin_centers[massmap]) 
    counts_sig=counts[massmap]-counts_bkg
    popt, pcov = curve_fit(gauss_function, bin_centers[massmap],counts_sig, p0 = [0, 2.9923, 0.0025])
    x=np.linspace(bin_centers[massmap][0]-0.01,bin_centers[massmap][-1]+0.01,100)
    x_bkg=np.polyval(h,x)
    plt.plot(x, gauss_function(x, *popt)+x_bkg,'b-',label='Gaussian model')
    plt.plot(bin_centers[massmap],counts[massmap],'r.',label='Signal region')
    plt.plot(bin_centers[sidemap],y,'g-',label='Sidebands fit')
    plt.plot(bin_centers[sidemap],counts[sidemap],'y.',label='Background rergion')
    tot_sig=sum(counts_sig)
    tot_bkg=sum(counts_bkg)

    plt.xlabel(r"$m_{\ ^{3}He+\pi^{-}}$")
    plt.ylabel(r"Events /  $3.6\ \rm{MeV}/c^{2}$") 
    plt.tick_params(axis="x", direction="in")
    plt.tick_params(axis="y", direction="in")  
    textstr =r"%1.f GeV/c $ \leq \rm{p}_{T} < $ %1.f GeV/c " %(pt[0],pt[1])
    props = dict(boxstyle='round',facecolor='white', alpha=0,)
    plt.title(textstr)
    plt.legend()
    print("Significance/Sqrt(Events) x 10^-4 = " , tot_sig/np.sqrt(tot_bkg+tot_sig)/np.sqrt(n_ev)*1e4)
    print("S/B = " ,tot_sig/tot_bkg)
    print("Raw yield = " , tot_sig)
