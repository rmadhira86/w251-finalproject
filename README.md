# W251 Final Project
## Manohar Madhira, Ryan Murphy, Napoleon Paxton,  Joe Villasenor

## Project Structure
The project structure is created based on `cookiecutter` folder structure for Machine Learning as documented [here](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa)   

The folder structure is as below:

|                           |                                                        |
|---------------------------|--------------------------------------------------------|    
|\|&nbsp;&nbsp;&nbsp;&nbsp;LICENSE                 |                                                        |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;Makefile                | <- Don't know how we will use this. Need to learn more.|  
|\|&nbsp;&nbsp;&nbsp;&nbsp;README.md               | This file. Will be updated with final project info     |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;data                    | Only folder structure will be on github. Data will be in local folders|  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;external           | Data from third party sources|  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;interim            | Intermediate data files that have been transformed |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;processed          | The final, canonical data sets for modeling |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;raw                | Raw data obtained by ourselves |  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;docs                       | Meant for Python module documents. We may not generate these|  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;models                     | For storing trained models and checkpoints |  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;notebooks                  | In case we use Jupyter Notebooks. Strongly recommend using Visual Studio Code with #%% magic that provides similar functionality as Notebooks without Markdown, but allows to write Python Script.  If used, naming convention is a number (for ordering), the creators initials, and a short `-` delimited description e.g. `1.0-rm-initial-data-exploration` |    
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;references | Data dictionaries, manuals, and all other explanatory materials |  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;reports                    | Generated analysis as HTML, PDF, LaTeX etc.|  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;figures           | Generated graphics and figures to be used in reporting|  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;requirements.txt                    | Requirements file generated using pipreqs |    
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;setup.py                    | Do not know how we will use this yet. Typically used to make this project pip installable with `pip install -e` |  
|                                |                                |
|\|&nbsp;&nbsp;&nbsp;&nbsp;src    | Source code used in this project - python scripts |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;__init.py__| Makes `src` a python module |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;data       | Scripts to download or generate data |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;features   | Scripts to turn raw data into features for modeling |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;models     | Scripts to train models and then used trained models for predictions |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;\|&nbsp;&nbsp;&nbsp;&nbsp;visualization | Scripts to create exploratory and results orientied visualizations |  
|\|&nbsp;&nbsp;&nbsp;&nbsp;tox.ini | Don't know how to use this yet. Need to learn from [tox.readthedocs.io](http://tox.readthedocs.io)  


