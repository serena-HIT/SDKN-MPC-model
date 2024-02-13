# SDKN-MPC-model
This is the open-source code for the SDKN-MPC nonlinear model predictive control system.

Firstly, ensure the correct versions of the following Python libraries:
Python 3.9.2, torch 2.1.0, scipy 1.8.0, numpy 1.22.4, matplotlib 3.8.0, pybullet 3.2.0, gym 0.23.1，and CUDA 12.3.

Before run the code, you should replace the gym env file with files in folder ./gym_env/

(1) All photos are stored in the 'pictures' folder.

(2) The raw data results for plotting are stored separately in the 'result' folder and under the 'control' folder in the 'model_compare_data' and 'short_predict_data' directories.

(3) If you want to rerun the plots, some paths in the code are absolute paths. Just change them to relative paths, and you'll be able to run the code. All data images in the paper can be obtained by running the code once.

(4) The 'stabler' folder under the 'control' directory contains various methods of SDKN-MPC. First, train the Koopman matrix and embedding functions using files starting with 'learn.' Then, run the files under the 'control' folder.

(5) The 'un_SOC' folder and the 'train' folder respectively store the control codes and training networks for DKUC, DKAC, and DKN without introducing the SOC algorithm. This is for comparison purposes.

(6) The 'compare' and 'SOC_compare' folders store codes for short-term and long-term prediction, as well as plotting codes. If you want to replicate the figures in the paper, you can run the code in these folders.
