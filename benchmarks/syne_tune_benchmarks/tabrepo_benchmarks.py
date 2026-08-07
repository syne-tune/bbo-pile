from benchmark_definitions import BenchmarkDefinition, n_full_evals

def tabrepo_benchmark(blackbox_name: str, dataset_name: str):
    return BenchmarkDefinition(
        max_wallclock_time=36000,
        max_num_evaluations=1 * n_full_evals,
        n_workers=1,
        elapsed_time_attr="metric_elapsed_time",  # todo should also include time_train_s + time_infer_s as metric
        metric="metric_error_val",  # could also do rank
        mode="min",
        blackbox_name=blackbox_name,
        dataset_name=dataset_name,
        use_surrogate=True,
        surrogate="KNeighborsRegressor",
        surrogate_kwargs={"n_neighbors": 1},
    )

tabrepo_search_spaces = [
    'tabrepo_RandomForest',
    'tabrepo_LinearModel',
    'tabrepo_CatBoost',
    'tabrepo_XGBoost',
    'tabrepo_ExtraTrees',
    'tabrepo_NeuralNetTorch',
    'tabrepo_LightGBM',
    'tabrepo_KNeighbors'
]

datasets = ['2dplanes', 'APSFailure', 'Airlines_DepDelay_10M', 'Allstate_Claims_Severity', 'Amazon_employee_access', 'Australian', 'Bioresponse', 'Brazilian_houses', 'Buzzinsocialmedia_Twitter', 'CIFAR_10', 'Click_prediction_small', 'Devnagari-Script', 'Diabetes130US', 'Fashion-MNIST', 'GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1', 'GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1', 'GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1', 'GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1', 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001', 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001', 'GTSRB-HOG01', 'GTSRB-HOG02', 'GTSRB-HOG03', 'GTSRB-HueHist', 'GesturePhaseSegmentationProcessed', 'Higgs', 'Indian_pines', 'Internet-Advertisements', 'KDDCup09-Upselling', 'KDDCup09_appetency', 'Kuzushiji-MNIST', 'LED-display-domain-7digit', 'MIP-2016-regression', 'MagicTelescope', 'Mercedes_Benz_Greener_Manufacturing', 'MiceProtein', 'MiniBooNE', 'Moneyball', 'OVA_Breast', 'OVA_Colon', 'OVA_Endometrium', 'OVA_Kidney', 'OVA_Lung', 'OVA_Ovary', 'OVA_Prostate', 'OnlineNewsPopularity', 'PhishingWebsites', 'QSAR-TID-10980', 'QSAR-TID-11', 'Run_or_walk_information', 'SAT11-HAND-runtime-regression', 'Santander_transaction_value', 'Satellite', 'SpeedDating', 'Titanic', 'Traffic_violations', 'UMIST_Faces_Cropped', 'Yolanda', 'abalone', 'ada', 'adult', 'ailerons', 'airlines', 'albert', 'analcatdata_authorship', 'analcatdata_dmft', 'anneal', 'arcene', 'arsenic-female-bladder', 'artificial-characters', 'autoUniv-au1-1000', 'autoUniv-au6-750', 'autoUniv-au7-1100', 'autoUniv-au7-700', 'balance-scale', 'bank-marketing', 'bank32nh', 'bank8FM', 'baseball', 'black_friday', 'blood-transfusion-service-center', 'boston', 'boston_corrected', 'car', 'cardiotocography', 'christine', 'churn', 'climate-model-simulation-crashes', 'cmc', 'cnae-9', 'colleges', 'colleges_usnews', 'collins', 'connect-4', 'covertype', 'cpu_act', 'cpu_small', 'credit-g', 'cylinder-bands', 'delta_ailerons', 'delta_elevators', 'diabetes', 'diamonds', 'dilbert', 'dna', 'dresses-sales', 'eating', 'eeg-eye-state', 'electricity', 'elevators', 'eucalyptus', 'eye_movements', 'fabert', 'fars', 'first-order-theorem-proving', 'fri_c0_1000_5', 'fri_c0_500_5', 'fri_c1_1000_50', 'fri_c2_1000_25', 'fri_c2_500_50', 'fri_c3_1000_10', 'fri_c3_1000_25', 'fri_c3_500_10', 'fri_c3_500_50', 'fri_c4_500_100', 'fried', 'gina', 'guillermo', 'har', 'helena', 'hill-valley', 'hiva_agnostic', 'house_16H', 'house_prices_nominal', 'house_sales', 'houses', 'hypothyroid', 'ilpd', 'isolet', 'jannis', 'jasmine', 'jm1', 'jungle_chess_2pcs_raw_endgame_complete', 'kc1', 'kc2', 'kdd_el_nino-small', 'kdd_internet_usage', 'kick', 'kin8nm', 'kr-vs-k', 'kropt', 'ldpa', 'led24', 'letter', 'madeline', 'madelon', 'mammography', 'mc1', 'meta', 'mfeat-factors', 'micro-mass', 'microaggregation2', 'mnist_784', 'mozilla4', 'no2', 'nomao', 'numerai28_6', 'nursery', 'nyc-taxi-green-dec-2016', 'okcupid-stem', 'one-hundred-plants-margin', 'optdigits', 'ozone-level-8hr', 'page-blocks', 'parity5_plus_5', 'pbcseq', 'pc1', 'pc2', 'pc3', 'pc4', 'pendigits', 'philippine', 'phoneme', 'pm10', 'pokerhand', 'pol', 'pollen', 'porto-seguro', 'puma32H', 'puma8NH', 'qsar-biodeg', 'quake', 'riccardo', 'ringnorm', 'rmftsa_ladata', 'robert', 'satimage', 'segment', 'semeion', 'sensory', 'sf-police-incidents', 'shuttle', 'socmob', 'soybean', 'space_ga', 'spambase', 'splice', 'spoken-arabic-digit', 'steel-plates-fault', 'sylvine', 'synthetic_control', 'tamilnadu-electricity', 'tecator', 'texture', 'tokyo1', 'topo_2_1', 'twonorm', 'us_crime', 'vehicle', 'visualizing_soil', 'volcanoes-a2', 'volcanoes-a3', 'volcanoes-a4', 'volcanoes-b1', 'volcanoes-b2', 'volcanoes-b5', 'volcanoes-b6', 'volcanoes-d1', 'volcanoes-d4', 'volcanoes-e1', 'volkert', 'walking-activity', 'wall-robot-navigation', 'waveform-5000', 'wilt', 'wind', 'wine-quality-red', 'wine-quality-white', 'wine_quality', 'yeast', 'yprop_4_1']

tabrepo_benchmark_definitions = {}

exclusion_list = ['tabrepo_KNeighbors_Amazon_employee_access',
                  'tabrepo_KNeighbors_GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1',
                  'tabrepo_KNeighbors_GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1',
                  'tabrepo_KNeighbors_GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1',
                  'tabrepo_KNeighbors_GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1',
                  'tabrepo_KNeighbors_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001',
                  'tabrepo_KNeighbors_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001',
                  'tabrepo_KNeighbors_KDDCup09-Upselling',
                  'tabrepo_KNeighbors_LED-display-domain-7digit',
                  "tabrepo_KNeighbors_Mercedes_Benz_Greener_Manufacturing",
                  'tabrepo_KNeighbors_PhishingWebsites',
                  'tabrepo_KNeighbors_QSAR-TID-10980',
                  'tabrepo_KNeighbors_QSAR-TID-11',
                  'tabrepo_KNeighbors_analcatdata_dmft',
                  'tabrepo_KNeighbors_autoUniv-au1-1000',
                  'tabrepo_KNeighbors_car',
                  'tabrepo_KNeighbors_connect-4',
                  'tabrepo_KNeighbors_dna',
                  'tabrepo_KNeighbors_hiva_agnostic',
                  'tabrepo_KNeighbors_kdd_internet_usage',
                  'tabrepo_KNeighbors_kropt',
                  'tabrepo_KNeighbors_led24',
                  'tabrepo_KNeighbors_nursery',
                  'tabrepo_KNeighbors_parity5_plus_5',
                  'tabrepo_KNeighbors_semeion',
                  'tabrepo_KNeighbors_sensory',
                  'tabrepo_KNeighbors_soybean',
                  'tabrepo_KNeighbors_splice',
                  ]
for ss in tabrepo_search_spaces:
    for ds in datasets:
        if ss + "_" + ds in exclusion_list:
            continue
        tabrepo_benchmark_definitions[ss + "_" + ds] = tabrepo_benchmark(ss, ds)