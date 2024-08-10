# Replication for ML Model

## Step 1: Prepare the Data
1. Unzip the training data:
   ```
   unzip bpp_dataset.zip
   ```
   This will create a folder named `bpp_dataset`.

## Step 2: Set Up the Environment
1. Change directory to the `bpp` folder:
   ```
   cd bpp
   ```
   This folder contains the training code.

2. Set up the Python environment:
   - Use the `environment.yml` file to create a conda environment that includes all necessary packages:
     ```
     conda env create -f environment.yml
     ```
   - Activate the environment:
     ```
     conda activate bpp
     ```

## Step 3: Train the Model
1. Run the training script with the specified parameters:
   ```
   python train.py --dim 16 --head 4 --transformer_layers 2 --rnn_layers 1 --seed 1234 --epochs 10000 --batch_size 512 --dataset bpp_dataset --multiply_a 10
   ```
   This script will create a new folder with a timestamp as its name, containing all run information.

## Step 4: Test the Model
1. Copy `test.ipynb` to the newly created timestamped folder.
2. Open and run the notebook cells in sequence to test the model's prediction accuracy.

## Step 5: Serialize the Model
1. Copy `to_torchjit.ipynb` to the timestamped folder.
2. Open and execute the notebook cells to serialize the model using `torch.jit`, generating a `.pt` file.




# Integrating ML Model with Neural Column Generation

## Step 6: Integrate the ML Model with NCG

1. Move the `.pt` file generated in Step 5 to the NCG folder.
2. Modify the `GlobalParams.json` file in the NCG folder. Specifically, change the `"machine_learning_model"` field to the name of your `.pt` file.

## Step 7: Run the NCG Algorithm

### Necessary Files
- `GlobalParams.json`: Configuration file for the algorithm. The default setting can be used to reproduce the results without any changes.
- `bpp_with_augment.pt`: The trained ML model in `.pt` format.
- DLL files required for running the executable `2L-CVRP-ML.exe`:
  - `pytorch_jni.dll`, `torch.dll`, `torch_cpu.dll`, `fbjni.dll`, `torch_global_deps.dll`, `c10.dll`, `fbgemm.dll`, `asmjit.dll`, `cublastLt64_10.dll`, `cublas64_10.dll`, `cufft64_10.dll`, `cusparse64_10.dll`, `curand64_10.dll`, `libiomp5md.dll`, `libiompstubs5md.dll`, `caffe2_detectron_ops.dll`, `caffe2_module_test_dynamic.dll`, `cplex2010.dll`.
   - We provide all the required DLL files except for `cplex2010.dll`. Due to licensing and copyright constraints, `cplex2010.dll` needs to be obtained separately by the user.

### Running the Executable
- The executable `2L-CVRP-ML.exe` is used to run the NCG algorithm.
- Download and place the required DLL files in the same directory as the executable.
- For `cplex2010.dll`, please visit the official website of IBM CPLEX Optimization Studio to download.


### Configuring `GlobalParams.json`
Here are some important fields in `GlobalParams.json` to experiment with:

1. `"ml_activate": true` - Set to `true` to activate the ML model; set to `false` to deactivate.
2. `"L_Trie": true` - `true` activates the L-Trie data structure; `false` deactivates it.
3. `"branch_bound": false` - `true` activates the branch-and-bound tree; `false` deactivates it, making it essentially the NCG approach as described in the paper.
4. `"InfSetCut": false` - `true` to generate infeasible set cuts; `false` to not generate them.

**Note:** It is advised to only modify the parameters mentioned above. Changing other parameters may lead to decreased performance or unexpected behavior of the algorithm.

## Citation  
You can find the arXiv version of the paper here: https://arxiv.org/abs/2406.12454.

> 🌟 If you find this resource helpful, please consider starting this repository and cite our research:

```bibtex
@inproceedings{ijcai2024p218,
  title     = {A Neural Column Generation Approach to the Vehicle Routing Problem with Two-Dimensional Loading and Last-In-First-Out Constraints},
  author    = {Xia, Yifan and Zhang, Xiangyi},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {1970--1978},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/218},
  url       = {https://doi.org/10.24963/ijcai.2024/218},
}

```
