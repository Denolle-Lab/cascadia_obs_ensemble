# Ensemble Deep Learning to further mine Cascadia offshore data


This project uses an ensemble deep learning algorith, ELEP (Yuan et al, 2023) to detect and pick P and S waves from continuous data in coastal and offshore Cascadia. It also uses PyOcto (Munchmeyer 2024) to associate/crudely locate events.

The repository also provides scripts to compare the new catalog with established catalog in the region (USGS ComCat, Stone et al 2018; Morton et al, 2023).

Project "CoolTeam" at UW:
- Hiroto Bito 
- Zoe Krauss
- Qibin Shi
- Yiyu Ni
- Marine Denolle

## Repository Structure
```

📜README.md
📜LICENSE
📜.gitignore
📜environment.yml
📦workflow
 ┣ 📜0_data_availability_7D.ipynb
 ┣ 📜1_parallel_detect_picks_elep.ipynb
 ┣ 📜2_format_pick2associate.ipynb
 ┣ 📜3_associate.ipynb
 ┗ 📜4_quality_control.ipynb
📦data
 ┣ 📜all_pick_assignments.csv
 ┣ 📜cat_elep.csv
 ┣ 📜ds01.csv
 ┣ 📜ds03.xlsx
 ┣ 📜jgrb52524-sup-0002-2017jb014966-ds01.csv
 ┗ 📜jgrb52524-sup-0003-2017jb014966-ds02.csv
📦plots
```