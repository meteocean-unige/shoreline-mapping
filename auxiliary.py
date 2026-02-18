import cv2
from datetime import datetime
import numpy as np
import os
from PIL import Image
from typing import List

from scipy.optimize import least_squares
from typing import Union

import matplotlib.pyplot as plt
import geopandas as gpd

from segment_anything import sam_model_registry, SamPredictor
import torch

from shapely.geometry import Point

import contextlib
import io
from io import StringIO

import pandas as pd
import re

#%% INPUT SETTING AND GEOMETRY
def read_set(filein):
    '''
    Imports general settings
    
    Input
    -----
    filein     : ASCII file setting.dat
        
    Output
    -----
    station_name  : CoastSnap Station
    start_date  : Initial date to be retainedz
    end_date  : Final date to be retained
    icut : min rate of points aligned as to retain picture 
    ic : pixel position for sea detection (column)
    ir : pixel position for sea detection (row)
    samP : type of predictor (vit_b (base) , vit_m (medium), vit_h (huge))
    '''

    with open(filein, "r") as file:
        lines = file.readlines()
    
    # only retain first column
    values = [line.split('%')[0].strip() for line in lines]

    # assign variables    
    start_date   = values[0]  # initial date (1st pic to process)
    end_date     = values[1]  # final date (last pic to process)    
    icut         = float(values[2])  # cut-off for retaining aligned pics
    ic   = int(values[3])     # column of the target pixel 
    ir   = int(values[4])     # raw of the target pixel 
    samP =  values[5]         # predictor for SAM    
    lon1 = float(values[6])
    lat1 = float(values[7])
    lon2 = float(values[8])
    lat2 = float(values[9])
    ds   = float(values[10])
            
    return start_date, end_date, icut, ic, ir, samP, lon1, lat1, lon2, lat2, ds

def read_gcp(filecrd):
    '''
    Imports general settings
    
    Input
    -----
    filein     : ASCII file GCP.dat
        
    Output
    -----
    v  : columns of the pixels corresponding to GCPs
    u  : rows of the pixels corresponding to GCPs    
    x  : UTM lon of the pixels corresponding to GCPs
    y  : UTM lat of the pixels corresponding to GCPs
    z  : elevation of the pixels corresponding to GCPs
    '''
        
    with open(filecrd, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]
    
    # store EPSG
    epsg_line = lines[0]
    epsg = int(epsg_line.split(":")[1])
    
    # store GCPs    

    header_index = next(i for i, line in enumerate(lines) if line.startswith("v"))
    data_lines = lines[header_index + 1 : ]
    data_str = "\n".join(data_lines)
            
    df = pd.read_csv(StringIO(data_str), sep=",", header=None, names=["v","u","x","y","z"])
        
    v = df["v"].to_numpy()
    u = df["u"].to_numpy()    
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    
    # detect GCPs
    gcps_index = next(i for i, line in enumerate(lines) if line.startswith("GCPs:"))
        
    gcps_line = lines[gcps_index]
    GCP = [int(n) for n in re.findall(r"\d+", gcps_line)]
    
    chars = ',][ '
    text = ''.join(str(GCP)) 
    for c in chars:
        text = text.replace(c, "")
        
    GCP = [g - 1 for g in GCP]
    
    return epsg, v[GCP], u[GCP], x[GCP], y[GCP], z[GCP], text

def sort_img_by_date(station_name,d1,d2):
    '''
    Extracts dates for each picture and retains those required by the user
    
    Input
    -----
    station_name     : main directory where pictures are stored
    d1     : Initial date 
    d2     : Final date
        
    Output
    -----
    img_sort  : List of pictures to be processed
    date_sort : Dates of the pictures to be processed
    '''
    
    DTO_EXIF_TAG = 0x9003 # TAG for date of origin in TAG
    img_date_dict = {}

    path_shared    = station_name + "/images"
    path_image_raw = path_shared + "/Raw/"

    # get list of the pictures
    imgs_to_align = os.listdir(path_image_raw)
    imgs_to_align = [img for img in imgs_to_align if img.lower().endswith(".jpg")
                     or img.lower().endswith(".png")
                     or img.lower().endswith(".jpeg")]             
        
    # loop through the pictures and extract dates
    for img in imgs_to_align:
        path_current_img = path_image_raw + img
    
        try:
            with Image.open(path_current_img) as img_PIL:
                exif_data = img_PIL._getexif()
    
            # if EXIF is missing --> cycle and move pictures to Unusable dir
            if not exif_data:
                print("No EXIF data found for:", img)
                print("Moving picture to Unusable folder\n")
                image_name = os.path.join(path_shared, "Unusable", img)
                os.replace(path_current_img, image_name)
                continue
    
            # extract date
            creation_time = exif_data.get(DTO_EXIF_TAG, None)
    
            # if date is missing --> cycle and move pictures to Unusable dir
            if creation_time is None:
                print("No EXIF date found for:", img)
                print("Moving picture to Unusable folder\n")
                image_name = os.path.join(path_shared, "Unusable", img)
                os.replace(path_current_img, image_name)
            else:
                img_date_dict[creation_time] = img
    
        except Exception as e:
            print(f"Error with {img}: {e}")
            print("Moving picture to Unusable folder\n")
            try:
                image_name = os.path.join(path_shared, "Unusable", img)
                os.replace(path_current_img, image_name)
            except:
                pass    
        
    # select pictures to be processed depending on dates        
    process_all = "Y" if d1 == "0" else "N"
    if process_all == "Y":            
        date_init = datetime.strptime('19760101', "%Y%m%d")
        date_end  = datetime.strptime('30000101', "%Y%m%d")
        print("[Retaining ALL pictures in RAW directory]\n")
            
    else:
        date_init = datetime.strptime(d1, "%Y%m%d")
        date_end = datetime.strptime(d2, "%Y%m%d")
        print("[Retaining pictures taken from " + str(date_init) +
                  " through " + str(date_end) + "]\n")
              
    date_to_process = []
    dateform  = "%Y:%m:%d %H:%M:%S"
    for date in img_date_dict:
        date = datetime.strptime(date, dateform)
        
        if date >= date_init and date <= date_end:
            date_to_process.append(date)
            
    date_sort = sorted(date_to_process)
    date_sort = [datetime.strftime(d,dateform) for d in date_sort]
    img_sort = [img_date_dict[date] for date in date_sort]

    return img_sort, date_sort

#%% ALIGNMENT
def find_matches(image_og: np.ndarray, image_target: np.ndarray):
    '''
    Finds matching keypoints between two similar images using feature detection
    
    Input
    -----
    image_og     : Image to be aligned
    image_target : Reference image
        
    Output
    -----
    matches           : List of matches beetween the pictures
    key_points_images : List of the respective keypoints beetween image_og 
                        and image_target
    '''
    
    image_target_gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
    image_gray        = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
    
    # [AKAZE] OPEN SOURCE
    akaze = cv2.AKAZE_create()
    (keypoints_target, desc_target) = akaze.detectAndCompute(image_target_gray,
                                                             None)
    (keypoints_og, desc_og) = akaze.detectAndCompute(image_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    # -------------------- ALTERNATIVES 4 OBJECT DETECTION --------------------
    # [ORB] OPEN SOURCE
    # orb = cv2.ORB_create()
    # (keypoints_target, desc_target) = orb.detectAndCompute(image_target_gray,
    #                                                        None)
    # (keypoints_og, desc_og) = orb.detectAndCompute(image_gray, None)
    # method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    # matcher = cv2.DescriptorMatcher_create(method)
    
    # [SIFT] OPEN SOURCE since recent version of opencv
    # sift = cv2.SIFT_create(nfeatures=20000)
    # (keypoints_target, desc_target)= sift.detectAndCompute(image_target_gray,
    #                                                         None)
    # (keypoints_og, desc_og) = sift.detectAndCompute(image_gray, None)    
    # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # -------------------------------------------------------------------------
    
    matches = matcher.match(desc_og,desc_target)
    matches = sorted(matches, key=lambda x:x.distance)
    
    # Filtration of the matches, currently 20%
    keep = int(len(matches) * 0.2)
    
    matches = matches[:keep]
    key_points_images = [keypoints_og,keypoints_target]
    
    return matches, key_points_images

KeypointsPair = List[List[cv2.KeyPoint]]
MatchesList = List[cv2.DMatch]

def alignment(image_og: np.ndarray,
               image_target: np.ndarray,
               key_points_images: KeypointsPair,
               matches: MatchesList 
               ):
    '''
    Aligns the reference onto the target using keypoint matches and homography
    
    Input
    -----
    image_og          : Image to aligned
    image_target      : Image model
    key_points_images : List of the respective keypoints beetween OG and TARGET
    matches           : List of matches beetween the pictures
        
    Output
    -----
    aligned : Aligned version of image_og
    '''
    
    keypoints_og     = key_points_images[0]
    keypoints_target = key_points_images[1]
    
    pts_target = np.zeros((len(matches), 2), dtype="float")
    pts_og     = np.zeros((len(matches), 2), dtype="float")
    
    for (i, m) in enumerate(matches):
        
        # indicate that the two keypoints in the respective images
        # map to each other
        pts_target[i] = keypoints_target[m.trainIdx].pt
        pts_og[i] = keypoints_og[m.queryIdx].pt
        
    (H, mask) = cv2.findHomography(pts_og, pts_target, method = cv2.RANSAC)
    (h, w)    = image_target.shape[:2]
    
    aligned = cv2.warpPerspective(image_og, H, (w, h))
    
    # compute error
    pts_og     = np.asarray(pts_og, dtype=np.float32)    
    projected = cv2.perspectiveTransform(pts_og.reshape(-1,1,2), H).reshape(-1,2)
    pts_target = np.asarray(pts_target, dtype=np.float32)
    dists     = np.linalg.norm(projected - pts_target, axis=1)   
    
    # enumerate successfull alignments
    mask    = mask.ravel().astype(bool)
    ALGpoints,ALLpoints = len(dists[mask==1]), len(dists)    
    algnmt_rate = ALGpoints/ALLpoints*100
    
    return aligned,algnmt_rate

#%% SEGMENTATION & SHORELINE EXTRACTION
def load_predictor(MODEL_TYPE):
    '''
    Loads predictor for Segment Anything Model algorithm
    
    # 3 different model :  vit_b (base) , vit_l (medium), vit_h (huge)
    '''
    path_model_sam = [f for f in os.listdir("./model_sam/") if MODEL_TYPE in f]
    path_model_sam = "model_sam/" + path_model_sam[0]
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Creation of SAM Model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=path_model_sam)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    return predictor

def is_black_area(image, u, v, threshold=10):
    region = image[max(u-2, 0):u+3, max(v-2, 0):v+3]

    # mean of RGB vals
    mean_val = np.mean(region)
    if mean_val < threshold:
        return False
    else:
        return True

def segment_image(image: np.ndarray,
                  predictor: SamPredictor,
                  target_col: int,
                  target_row: int):
    '''
    Segments the image using SAM from a single point
    
    Input
    -----
    image     : Image to segment
    predictor : Model used to predict the mask
    ir        : Ratio value for the input point (y)
    ic        : Ratio value for the input point (x)
    
    Output
    -----
    masks       : The predicted segmentation masks
    input_point : Point used for the segmentation
    '''
    
    input_point = np.array([[target_col,target_row]])
    input_label = np.array([1])
    
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,)

    mask = masks[0]
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    contours, _  = cv2.findContours(mask_uint8,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    
    return mask, contours

#%% SHORELINE PROJECTION
def normalize_points(points: np.ndarray):
    """
    Normalize 2D homogeneous points so they have zero centroid and unit average
    distance to the origin.
    
    This normalization improves numerical stability when estimating camera 
    projection matrices using DLT
    
    Input
    -----
    points : Input points as a 2xN or 3xN array. 
             Only the first two rows (x, y) are used for normalization.
    
    Output
    -----
    normalized_points : The normalized points as a 3xN array
    The 3x3 normalization transformation matrix applied to the original points
    """
    
    assert points.shape[0] in (2, 3)
    
    if points.shape[0] == 2:
        points = np.vstack((points, np.ones((1, points.shape[1]))))
    
    points = points.astype(np.float64)

    # Compute centroid of the (x, y) coordinates
    centroid = np.mean(points[:2], axis=1)
    shifted = points[:2] - centroid[:, None]
    
    # Compute root mean square distance to the origin
    rms = np.sqrt(np.mean(np.sum(shifted**2, axis=0)))
    
    # Compute scaling factor so that RMS = sqrt(2)
    scale = np.sqrt(2) / (rms + 1e-12)

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    normalized_points = T @ points
    return normalized_points, T

def dlt_matrix(world_points: np.ndarray, 
               image_points: np.ndarray):
    """
    Construct the linear system matrix A used in the Direct Linear Transform

    Input
    -----
    world_points  : 3xN array of 3D world points in the form [X, Y, Z] 
    image_points  : 3xN array of 2D image points in homogeneous coordinates 
                    [u, v, 1]
        
    Output
    -----
    A : The 2N x 12 matrix used in the homogeneous linear system A * p = 0
    """
    
    assert world_points.shape[0] == 3
    assert image_points.shape[0] == 3
    
    n = world_points.shape[1]
    A = []
    
    for i in range(n):
        X, Y, Z = world_points[:, i]
        u, v = image_points[0, i], image_points[1, i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
        
    return np.asarray(A, dtype=np.float64)

def dlt_algo(world_points: np.ndarray,
             image_points: np.ndarray):
    """
    Estimate the 3x4 camera projection matrix P using the Direct Linear 
    Transform (DLT) algorithm.
    
    Steps:
    1. Normalize the image points for better numerical stability.
    2. Build the DLT matrix A from the 3D-2D correspondences.
    3. Solve A * p = 0 using Singular Value Decomposition (SVD).
    4. Reshape the solution vector p into a 3x4 projection matrix.
    5. Denormalize the projection matrix to return it to the original 
       coordinate system.
    
    Input
    ----
    world_points : Array of 3D world points of shape (3, N)
    image_points : Array of 2D image points of shape (2, N)
    
    Output
    ----
    P : The 3x4 camera projection matrix
    """
    
    # Normalize input pts
    norm_img_pts, T = normalize_points(image_points)
    
    # Build the  matrix A
    A = dlt_matrix(world_points.astype(np.float64), norm_img_pts)
    
    # Solve A * p = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    P_norm = Vt[-1].reshape(3, 4)
    
    # Denormalize (return to its original coordinate system)
    P = np.linalg.inv(T) @ P_norm
    
    # Normalize the result
    denom = P[-1, -1] if abs(P[-1, -1]) > 1e-12 else np.linalg.norm(P)
    P = P / denom
    return P

def project_points(P_vec: Union[np.ndarray, list],
                   world_points: np.ndarray):
    """
    Projects 3D world points onto the image plane using a camera projection matrix.
    
    Input
    ----
    P_vec        : Flattened camera projection matrix N=11 or 12
    world_points : 3xN array of 3D world points
    
    Output
    ----
    projected_points : 2xN array of 2D projected points
    """
    
    # Homogenize the projection matrix vector by appending 1
    if len(P_vec) == 11:
        P_vec = np.append(P_vec, 1.0)
    
    P = P_vec.reshape(3, 4)
    
    # Convert world points to homogeneous coordinates: (4xN)
    world_hom = np.vstack((world_points, np.ones((1, world_points.shape[1]))))
    proj = P @ world_hom

    # Apply projection: (3xN)
    z = proj[2]
    z[np.abs(z) < 1e-12] = 1e-12
    proj /= z
    
    return proj[:2]

def reprojection_residuals(P_vec: Union[np.ndarray, list],
                           world_points: np.ndarray,
                           image_points: np.ndarray):
    """
    Compute the reprojection residuals between projected 3D world points
    and their observed 2D image points. 
    
    Used for the optimization routine to minimize the reprojection error
    
    Input
    -----
    P_vec        : Flattened 3x4 projection matrix (length 11 or 12)
    world_points : 3D world coordinates, shape (3, N)
    image_points : 2D image coordinates, shape (2 or 3, N)
    
    Output
    ------
    residuals : Flattened vector of reprojection errors, shape (2*N,)
    """
    
    # Get the result of the projection
    projected = project_points(P_vec, world_points)
    
    # Ensure image points are 2D
    img2 = image_points[:2].astype(np.float64)
    
    # Compute residuals and flatten to 1D vector (for least_squares)
    return (projected - img2).ravel()

def refine_projection_matrix_robust(P_init: np.ndarray, 
                                    world_points: np.ndarray, 
                                    image_points: np.ndarray,
                                    loss: str = 'soft_l1', 
                                    f_scale: float = 1.0, 
                                    method: str = 'trf', 
                                    verbose: int = 0):
    """
    Refine a camera projection matrix using robust non-linear least squares 
    optimization.

    It uses a robust loss function to reduce the influence of outliers and 
    returns the optimized projection matrix.

    Input
    -----
    P_init       : Initial estimate of the 3x4 projection matrix
    world_points : 3D world coordinates, shape (3, N)
    image_points : 2D image coordinates, shape (2 or 3, N)
    loss         : Loss function used to reduce outlier sensitivity 
                  ('linear', 'soft_l1', 'huber', 'cauchy')
    f_scale      : Scaling parameter for the loss function
    method       : Optimization method ('trf' or 'dogbox') 
                   — 'lm' is not supported for robust loss
    verbose      : Verbosity level for optimization 
                   (0 = silent, 2 = full output)

    Output
    ------
    P_opt : Optimized 3x4 projection matrix
    res   : Optimization result - object from scipy.optimize.least_squares
    """
    
    # Remove the last value (assumed to be 1)
    P_vec_init = P_init.ravel()[:-1]
    
    res = least_squares(reprojection_residuals,
                        P_vec_init,
                        args=(world_points.astype(np.float64), 
                              image_points.astype(np.float64)),
                        method = method,
                        loss = loss,
                        f_scale = f_scale,
                        max_nfev = 2000,
                        verbose = verbose)
    
    P_opt = np.append(res.x, 1.0).reshape(3, 4)
    return P_opt, res

def compute_reprojection_error(P: np.ndarray, 
                               world_points: np.ndarray, 
                               image_points: np.ndarray):
    """
    Compute reprojection error per point.

    Input
    -----
    P            : (3, 4) projection matrix
    world_points : (3, N) world coordinates (x, y, z)
    image_points : (2 or 3, N) image coordinates (u, v)

    Output
    ------
    errors       : (N,) reprojection error in pixels for each point
    """
    # Homogénéisation des points monde
    X_hom = np.vstack((world_points, np.ones((1, world_points.shape[1]))))  # 4xN
    proj = P @ X_hom  # 3xN
    proj /= proj[2]  # homogénéisation

    # Erreur en pixels (euclidienne)
    img2 = image_points[:2].astype(np.float64)
    error = np.linalg.norm(proj[:2] - img2, axis=0)  # (N,)
    return error

def camera_projection(world_pts: np.ndarray,
                      image_pts: np.ndarray):
    """
    Estimate and refine the camera projection matrix using DLT followed by
    robust optimization.
    
    Input
    -----
    world_pts : 3D world coordinates, shape (3, N)
    image_pts : 2D image coordinates, shape (2 or 3, N)

    Output
    -----
    P0 : Initial projection matrix estimated via DLT (3x4)
    P_opt : Refined projection matrix after optimization (3x4)
    res : Optimization result object returned by scipy.optimize.least_squares
    """
    
    P0 = dlt_algo(world_pts, image_pts)
    P_opt, res = refine_projection_matrix_robust(P0,
                                                 world_pts, image_pts,
                                                 loss='soft_l1', #loss='soft_l1'
                                                 f_scale=1.0, 
                                                 method='trf', 
                                                 verbose=2)
    
    print(P_opt)
    err = compute_reprojection_error(P_opt, world_pts, image_pts)
    
    print("Reprojection error per point (pixels):")
    print(np.round(err, 2))  # Arrondi pour lisibilité
    
    print(f"\nMean error: {np.mean(err):.2f} px")
    print(f"Max error: {np.max(err):.2f} px")
    return P0, P_opt, res

def camera_proj(v, u, x, y, z):
    
    # compute camera projection matrix
    xyz      = np.vstack((x, y, z))    
    nmx, nmy = x - np.mean(x), y - np.mean(y)
    mxyz     = np.vstack((nmx,nmy,xyz[2]))    
    
    vu = np.vstack((v,u))
    
    A_ , A, res = camera_projection(mxyz, vu)

    return A_ , A, res
    
def map_utm(A, vw, uw, zc: float=0):
    '''
    Projects 2D image pixel coordinates into 2D UTM world coordinates
    
    Input
    -----
    A  : Camera projection matrix [3x4]
    uw : Horizontal pixel coordinates (x axis)
    vw : Vertical pixel coordinates (y-axis)
    zc : Elevation (0 by default for sea level)
        
    Output
    -----
    world_pts_est : Coordinates containing the estimated Lat, Long UTM
    '''
    
    vw = np.asarray(vw)
    uw = np.asarray(uw)    
    N = len(uw)

    A3 = A[:, 2]
    Ar = A[:, [0, 1, 3]]

    world_pts_est = np.zeros((N, 2))

    for i in range(N):
        uvH = np.array([vw[i], uw[i], 1])
        rhs = uvH - A3 * zc  
        xyH = np.linalg.solve(Ar, rhs)
        xy = xyH[:2] / xyH[2]     
        world_pts_est[i, :] = xy
            
    return world_pts_est

#%% LAUNCH SCRIPT
def process_pic(station_name):
    '''
    Main function aligning all the picture on one target image
    
    Input
    -----
    filein : ASCII file with general setting

    '''    
    
    # define paths 
    filein  = station_name + '/setting.dat'
    filecrd = station_name + '/GCP.dat'
    
    # load general settings and crds of the GCPs        
    dateINI, dateEND, thr_al, target_col, target_row, samP, lon1, lat1, lon2, lat2, ds = read_set(filein)
    epsg, v, u, x, y, z, filend = read_gcp(filecrd)

    # get list of pictures to be processed
    img_sort, date_sort = sort_img_by_date(station_name,dateINI,dateEND)
    
    # define paths
    path_shared        = station_name + "/images"
    path_image_raw     = path_shared + "/Raw/"
    path_image_out     = path_shared + "/Processed/"
    path_image_target  = path_shared + "/Target/"
    path_shape_out     = station_name + "/shorelines/"
        
    path_current_target = os.listdir(path_image_target)[0]
    path_current_target = path_image_target + path_current_target
    
    image_target = cv2.imread(path_current_target)
    
    # define SAM model
    predictor = load_predictor(samP)    
    
    # compute camera projection matrix
    f = io.StringIO()  
    with contextlib.redirect_stdout(f): 
        A_, A, res = camera_proj(v, u, x, y, z)
        
    for i in range(len(img_sort)):  
        img = img_sort[i]
        creation_time = date_sort[i]
        creation_time = creation_time.replace(":","_").replace(" ","-")

        print(f"Processing pic {i+1}/{len(img_sort)} | Name: {img} | Taken: {creation_time}")
        
        path_current_img = path_image_raw + img
        image_out = (path_image_out + creation_time + ".png")
        
        # load raw picture and extract matching objects
        image_og  = cv2.imread(path_current_img)        
        (matches, key_points_images) = find_matches(image_og, image_target)

        # Do the homography and the alignement of the image
        image_aligned,algnmt_rate = alignment(image_og, image_target,
                                                   key_points_images, matches)  
        
        # get back original colors and shape
        image = cv2.cvtColor(image_aligned, cv2.COLOR_BGR2RGB)

        # check if target pixel lies in the border ...        
        target_valid = is_black_area(image,target_row,target_col)

        # ... if not, do the segmentation
        if target_valid and algnmt_rate>thr_al:
            
            mask, contours = segment_image(image,predictor,target_col,target_row)        

            # get shoreline pixels
            cnt = max(contours, key=cv2.contourArea)
            coords = cnt.squeeze()
            vc = coords[:, 0]  # columns
            uc = coords[:, 1]  # rows
            
            plt.figure()
            plt.imshow(image)
            plt.imshow(mask, alpha=0.4, cmap='gray')            
            plt.plot(vc,uc,'r-',linewidth=1)
            plt.axis('off')            
            plt.savefig(image_out, bbox_inches='tight', pad_inches=0, dpi=250)
                               
            # project shoreline in UTM coord
            world_pts_est = map_utm(A, vc, uc)
            
            lon = np.add(world_pts_est[:, 0], np.mean(x))
            lat = np.add(world_pts_est[:, 1], np.mean(y))            

            # slice coastline to selected area
            mask = ((lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2))
            lon_box, lat_box = lon[mask], lat[mask]         

            # interp to desired resolution   
            deltas = np.sqrt(np.diff(lon_box)**2 + np.diff(lat_box)**2)            
            s      = np.insert(np.cumsum(deltas), 0, 0)  # distance along line [m]
            s_uniform = np.arange(0, s[-1], ds) 

            lon_intrp = np.interp(s_uniform, s, lon_box)
            lat_intrp = np.interp(s_uniform, s, lat_box)
                        
            # # make it convex
            # coords  = list(zip(lon,lat))
            # mp      = MultiPoint(coords)
            # mp_conv = mp.convex_hull
            
            # clon, clat = mp_conv.exterior.xy
            # clon, clat = np.asarray(clon), np.asarray(clat)
                    
            # export data to shapefile
            lon_all = np.concatenate([lon, lon_intrp])
            lat_all = np.concatenate([lat, lat_intrp])

            lbl = (
                ["sam"] * len(lon) +
                ["interpolated"] * len(lon_intrp)
                )

            geometry = [Point(xy) for xy in zip(lon_all, lat_all)]

            gdf = gpd.GeoDataFrame(
                {"type": lbl},
                geometry=geometry,
                crs="EPSG:" + str(epsg)
                )

            gdf.to_file(path_shape_out + creation_time + "_" + filend + ".shp")

        else:
            
            print("Alignment and/or SAM failed")
            print("Moving picture to Unusable folder\n")
            image_name = os.path.join(path_shared, "Unusable", img)
            os.replace(path_current_img, image_name)                
               
            del image_og        
            del image_aligned
        
    log_string = f"Last processed: {img} | Taken: {creation_time}\n"
    filelog    = station_name + "/processing.log"
    with open(filelog, "a", encoding="utf-8") as f:
        f.write(log_string)
  
