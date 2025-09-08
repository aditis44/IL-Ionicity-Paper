Purpose: Code used for building ML models to predict the ionicity & molar ionic conductivity of ionic liquids & analyzing results

Overall Approach:
- Develop ML models for predicting the molar conductivity & ionicity of ionic liquids, using data from the NIST ILThermo database
    - Train linear (L1 regularization),and XGBoost models, tune hyperparameters, track results using Weights & Biases (https://wandb.ai/site/)
    - Train models using input features: RDKit descriptors, sigma profile-based descriptors, and RDKit + sigma profile-based descriptors
- Analysis:
    - Compare the model performance & feature importance ranking across: ML method (linear, XGBoost), input features to the model (RDKit descriptors, sigma profile-based descriptors)

Files/Folders:
- base directory: 
    - createPlotsML.py - code used to create parity plots, calculate model performance for each model/set of results
    - ionic_liquid_SMILES_functions.py - code used to manipulate/obtain information from ionic liquid SMILES strings or cation/anion SMILES strings
    - train_models.py - code used to train models to predict ionicity or molar conductivity, and tune hyperparameters
    - feature_importance_analysis.py - code used for feature importance analysis
    - ref_train_models_cv.sh - example job submission script for 5-fold cross-validation using S_i input descriptors
    - ref_train_eval_optimal_models.sh - example job submission script for training models w/optimal hyperparameters & obtaining feature importance rankings using S_i input descriptors
- sigma_profile_calculations folder: code & reference job submission scripts used for calculating sigma profiles using openCOSMO-RS & ORCA
- ILThermo_Preprocessed_Data: csv files with preprocessed dataset used for training models & dataset analysis (includes for each IL the compound name, SMILES strings, T,P, viscosity, density, ionic conductivity, estimated ionic radii and ionicity along with different input descriptors - RDKit, sigma profiles, etc.)
    - in ILThermo_Preprocessed_Data/ILThermo_Preprocessed_Data_Splits folder, there are 3 main types of csv files  
        - x: corresponds to x data/features that are used as input to the model (ex: RDKit descriptors, sigma profile-based descriptors)
        - y: corresponds to the y data/labels that are used when training & evaluating the model (i.e. molar ionic conductivity or ionicity)
        - dataset_info: corresponds to any additional information for the ILs that is not in the x or y csv files

Python package versions:
- numpy              1.26.4
- rdkit              2024.3.5
- scikit-learn       1.5.2
- xgboost            2.1.3
- shap               0.48.0
- wandb              0.20.1
- pandas             2.2.2
- matplotlib         3.9.0
- seaborn            0.13.2
- PubChemPy          1.0.4
- mordred            1.2.0

For sigma profile calculations, ORCA 6.0.0 was used along with the openCOSMO-RS conformer pipeline (https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline)

General References: (additional references specific to parts of the code are provided where they were used)
Papers on ML in materials science/chemistry, best practices, etc
    - Wang, A. Y. T., Murdock, R. J., Kauwe, S. K., Oliynyk, A. O., Gurlo, A., Brgoch, J., ... & Sparks, T. D. (2020). Machine learning for materials scientists: an introductory guide toward best practices. Chemistry of Materials, 32(12), 4954-4965.
        - GitHub Repository associated with this paper: https://github.com/anthony-wang/BestPractices
    - Packwood, D., Nguyen, L. T. H., Cesana, P., Zhang, G., Staykov, A., Fukumoto, Y., & Nguyen, D. H. (2022). Machine learning in materials chemistry: An invitation. Machine Learning with Applications, 8, 100265.

NIST ILThermo dataset: https://www.nist.gov/mml/acmd/trc/ionic-liquids-database
    - Kazakov, A.; Magee, J.W.; Chirico, R.D.; Paulechka, E.; Diky, V.; Muzny, C.D.; Kroenlein, K.; Frenkel, M. "NIST Standard Reference Database 147: NIST Ionic Liquids Database - (ILThermo)", Version 2.0, National Institute of Standards and Technology, Gaithersburg MD, 20899, http://ilthermo.boulder.nist.gov.
    - Dong, Q.; Muzny, C.D.; Kazakov, A.; Diky, V.; Magee, J.W.; Widegren, J.A.; Chirico, R.D.; Marsh, K.N.; Frenkel, M., "ILThermo: A Free-Access Web Database for Thermodynamic Properties of Ionic Liquids." J. Chem. Eng. Data, 2007, 52(4), 1151-1159, doi: 10.1021/je700171f.

ILThermoPy Python package used to analyze & interface w/the ILThermo database:
    - https://github.com/IvanChernyshov/ILThermoPy
    - https://ilthermopy.readthedocs.io/en/latest/

openCOSMO-RS
    - Müller, S., Nevolianis, T., Garcia-Ratés, M., Riplinger, C., Leonhard, K., & Smirnova, I. (2025). Predicting solvation free energies for neutral molecules in any solvent with openCOSMO-RS. Fluid Phase Equilibria, 589, 114250.
    - https://github.com/TUHH-TVT/openCOSMO-RS_conformer_pipeline

ORCA 6.0.0
    -  Neese,F. Software update: the ORCA program system, version 5.0. WIRES Comput. Molec. Sci., 2022 12(1)e1606. doi.org/10.1002/wcms.1606

Weights & Biases package/platform used to track ML model results when varying hyperparameters:
    - Biewald, L. (2020). Experiment Tracking with Weights and Biases.
    - https://wandb.ai/site/research/

Some papers relating to ionicity, ion transport, ionic conductivity and ionic liquids (NOT exhaustive):
    - Watanabe, M. (2016). Design and materialization of ionic liquids based on an understanding of their fundamental properties. Electrochemistry, 84(9), 642-653.
    - Hollóczki, O., Malberg, F., Welton, T., & Kirchner, B. (2014). On the origin of ionicity in ionic liquids. Ion pairing versus charge transfer. Physical Chemistry Chemical Physics, 16(32), 16880-16890.
    - Nordness, O., & Brennecke, J. F. (2020). Ion dissociation in ionic liquids and ionic liquid solutions. Chemical reviews, 120(23), 12873-12902.
    - Umaña, J. E., Cashen, R. K., Zavala, V. M., & Gebbie, M. A. (2025). Uncovering ion transport mechanisms in ionic liquids using data science. Digital Discovery, 4(6), 1423-1436.
        - GitHub Repository associated with this paper: https://github.com/zavalab/ML/tree/master/IonicLiquids
    - Cashen, R. K., Donoghue, M. M., Schmeiser, A. J., & Gebbie, M. A. (2022). Bridging database and experimental analysis to reveal super-hydrodynamic conductivity scaling regimes in ionic liquids. The Journal of Physical Chemistry B, 126(32), 6039-6051.
    - Dhakal, P., & Shah, J. K. (2022). A generalized machine learning model for predicting ionic conductivity of ionic liquids. Molecular Systems Design & Engineering, 7(10), 1344-1353.
    - Dhakal, P., & Shah, J. K. (2021). Developing machine learning models for ionic conductivity of imidazolium-based ionic liquids. Fluid Phase Equilibria, 549, 113208.
        - GitHub Repository associated with this paper: https://github.com/ShahResearchGroup/Machine-Learning-Model-for-Imidazolium-Ionic-Liquids
