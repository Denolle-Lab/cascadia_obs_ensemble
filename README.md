# Ensemble Deep Learning to further mine Cascadia offshore data


This project uses an ensemble deep learning algorithm, ELEP (Yuan et al, 2023) to detect and pick P and S waves from continuous data in coastal and offshore Cascadia using the 2011-2015 Cascadia Initiative Experiment Ocean Bottom Seismometers.

The workflow also includes association, location, and relocation using : PyOcto (Munchmeyer 2024), HypoInverse (Klein 2002), and HypoDD (Waldhauser and Ellsworth 2000) for relocation.

The repository also provides scripts to compare the new catalog with established catalog in the region ([USGS ComCat](https://earthquake.usgs.gov/earthquakes/search/) and [Morton et al, 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JB026607)).

Project "CoolTeam" at UW:
- Hiroto Bito: hbito@uw.edu 
- Zoe Krauss: zkrauss@uw.edu
- Qibin Shi: qibins@uw.edu
- Yiyu Ni: niyiyu@uw.edu
- Marine Denolle: mdenolle@uw.edu
- Nate Stevents: ntstevens@uw.edu

Project team at Stanford:
- Ian MacBreatry

## Repository Structure
```

📜README.md
📜LICENSE
📜.gitignore
📜environment.yml
📦0_availability
📦1_picking
📦2_associate
📦workflow_simplified
 ┣ 📜0_data_availability_7D.ipynb
 ┣ 📜1_parallel_detect_picks_elep.ipynb
 ┣ 📜2_format_pick2associate.ipynb
 ┣ 📜3_associate.ipynb
 ┗ 📜4_quality_control.ipynb
📦data
 ┣ 📜vel_*.csv # velocity profiles
 ┣ 📜stations_*.csv # station locations in regions
 ┣ 📜nodes_*.csv # regions for velocity profiles
 ┣ 📜ds03.xlsx
 ┣ 📜jgrb52524-sup-0002-2017jb014966-ds01.csv # Morton et al, 2023 data
 ┗ 📜jgrb52524-sup-0003-2017jb014966-ds02.csv # Morton et al, 2023 data
📦plots
```

# Installation Guide

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- [Python 3.8+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

## Setting Up the Environment

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Denolle-lab/cascadia_obs_ensemble.git
    cd cascadia_obs_ensemble
    ```

2. **Create a virtual environment:**

    ```sh
    conda env create -f environment.yml
    ```

3. **Activate the virtual environment:**

    - On macOS and Linux:

        ```sh
        conda activate seismo_cobs
        ```

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

4. **Install the required packages:**

    If you have a [requirements.txt](http://_vscodecontentref_/0) file:

    ```sh
    pip install -r requirements.txt
    ```

    If you have an [environment.yml](http://_vscodecontentref_/1) file (for Conda environments):

    ```sh
    conda env create -f environment.yml
    conda activate your-environment-name
    ```

## Running the Notebooks

1. **Start Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open the notebooks:**

    In the Jupyter Notebook interface, navigate to the following notebooks and open them:
    <!-- ```
    - ./workflow_simplified/3_associate.ipynb
    - ./workflow_simplified/4_quality_control.ipynb
    ``` -->

## Pick data

After running ELEP, the picks are now available on Gdrive (private access until we upload to zenodo)

* [Picks](https://drive.google.com/drive/folders/1ACsaRj3GY-kBwPoXGb-RCDAlEiM3ArJP)


<!-- * [Picks 2011](https://drive.google.com/file/d/1D2hXbtvMPTiktmcg_ugrvIz8W_82x5Ht/view?usp=drive_link)
* [Picks 2012](https://drive.google.com/file/d/1gq2isg0dOuaRmorQRAc5PJKstBEjFd5L/view?usp=drive_link)
* [Picks 2013](https://drive.google.com/file/d/1M8UNhKxewNG48Rsjnk_DbWl2NpsgXXp8/view?usp=drive_link)
* [Picks 2014](https://drive.google.com/file/d/1sV7yTBDfVhBUixA0NvCmZKyf1L1SJh-C/view?usp=drive_link)
* [Picks 2015](https://drive.google.com/file/d/15Ok11F3r2Ia-5KanmlMGCDOhL0Usckpr/view?usp=drive_link)
 -->