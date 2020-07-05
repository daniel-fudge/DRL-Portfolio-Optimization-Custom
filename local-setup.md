# Local PC setup
The following instructions are written for a Windows environment but should be extremely similar for Linux or OS X.

1. Clone repo to PC.  If you don't have git installed, you can get it [here](https://git-scm.com/).
   ```shell script
   git clone https://github.com/daniel-fudge/DRL-Portfolio-Optimization-Custom.git
   ```

1. Install [Anaconda3](https://www.anaconda.com/distribution/).

1. Create a virtual `customdrl` environment.  Please see this [site](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
for more conda environment information.  If you need to delete the `customdrl` environment type: `conda env remove --name customdrl`.
To deactivate the environment type: `conda deactivate`.

   ```shell script
   conda update -n base -c defaults conda -y
   conda clean -tipsy
   conda env create -f environment.yml
   conda activate customdrl
   python -m ipykernel install --user --name customdrl --display-name "Custom DRL"
   ```
   
1. Activate the virtual environment if not already activated.  Note the above steps only need to be executed once.  
After initial setup you can start at this step.
   ```shell script
   conda activate customdrl
   ```

1. Launch Jupyter.  
_**Note**: Launching jupyter from a Windows PowerShell may not work. I recommend using the standard
command prompt._
   ```shell script
   jupyter notebook
   ```

1. Open the desired notebook.

1. Read and/or run the cells at your leisure!!  :)
