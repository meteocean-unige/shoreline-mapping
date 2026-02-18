# shoreline-mapping
Python-based tools for extracting georeferenced shoreline data from crowdsourced photographs.

De Leo, F., Pineau, A., Lisboa, D., & Besio, G. (2025)

The working directory must contain the following elements:

1. Python Scripts
  
defPixels.py – Used for the initial selection of the Ground Control Points (GCPs).  
  
auxiliary.py – Library used to:  
    - Align raw images to the target image;  
    - Segment the aligned images;  
    - Project the extracted shorelines onto a UTM coordinate system.  
  
main.py – Main script used to launch the entire processing workflow. The input argument must match the name of the monitoring station (e.g., sturla in the provided example).  

coastsnap_env.yml is the environment required to run the scripts.  

2. model_sam Folder

A folder named model_sam containing the checkpoints of the Segment Anything Model (SAM).
The model weights can be downloaded from:
https://github.com/facebookresearch/segment-anything

3. Station Folder

A folder named after the monitoring station (e.g., sturla).
This folder must contain:
  - GCP.dat – ASCII file containing the coordinates of the Ground Control Points (GCPs), both in pixel coordinates and in the UTM reference system (see attached example);
  - setting.dat – ASCII file containing general configuration settings. All entries are commented and self-explanatory (see attached example).

Subfolders within the Station Directory:
  - shorelines --> contains the extracted shorelines saved in shapefile format (UTM coordinates);
  - images --> this folder must contain four subfolders:
      - Target – Contains the reference image used to align all raw images (see attached example);
      - Raw – Contains all raw images to be processed (see attached example);
      - Unusable – Images for which alignment fails or metadata cannot be accessed are automatically moved to this folder;
      - Processed – Contains raw images after alignment and segmentation.

The scripts were developed building upon the CoastSnap framework. For reference see:
Harley, M. D., & Kinsela, M. A. (2022). CoastSnap: A global citizen science program to monitor changing coastlines. Continental Shelf Research, 245, 104796.

