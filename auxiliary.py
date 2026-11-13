import cv2
from datetime import datetime, timedelta
import numpy as np
import os
from PIL import Image
from typing import List, Optional, Tuple, Dict

from dlt_v2 import camera_projection
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

from segment_anything import sam_model_registry, SamPredictor
import torch

import contextlib
import io
from io import StringIO

import pandas as pd
import re

# --- people counting (see PEOPLE COUNTING SETTINGS cell) ---------------------
import csv
from ultralytics import YOLO


# If True, when EXIF date is missing the date is guessed from the file name,
# but ONLY if it does not coincide with the download date of the file.
ALLOW_FILENAME_DATE = True

# Filesystem timestamps compared against the file-name date.
#   "all"   -> reject if the file-name date matches ANY of birth/ctime/mtime
#   "birth" -> reject only if it matches the creation (download) timestamp
DOWNLOAD_DATE_MODE = "all"

# Accepted image extensions
IMG_EXT = (".jpg", ".jpeg", ".png")

# Plausible range for a CoastSnap picture
DATE_MIN = datetime(1990, 1, 1)

#%% INPUT SETTING AND GEOMETRY
def read_set(filein):
    '''
    Imports general settings

    Input
    -----
    filein     : ASCII file setting_<season>.dat

    Output
    -----
    station_name  : CoastSnap Station
    start_date  : Initial date to be retained
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
    values = [v for v in values if v != ""]

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
    Imports GCPs

    Input
    -----
    filecrd    : ASCII file GCP_<season>.dat

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

#%% SEASONS (MULTI-TARGET SUPPORT)

def read_seasons(filein):
    '''
    Imports the season definition

    Expected format of seasons.dat (one season per line):

        cold, 10,11,12,1,2,3    % winter target
        warm, 4,5,6,7,8,9       % summer target

    Input
    -----
    filein : ASCII file seasons.dat

    Output
    -----
    month2season : dict {month (1-12) : season name}
    season_names : list of season names, in the order they were declared
    '''

    month2season = {}
    season_names = []

    with open(filein, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.split('%')[0].strip()
        if line == "":
            continue

        parts = [p.strip() for p in line.split(',') if p.strip() != ""]
        name   = parts[0]
        months = [int(m) for m in parts[1:]]

        if name in season_names:
            raise ValueError(f"[seasons.dat] duplicated season name: {name}")
        if len(months) == 0:
            raise ValueError(f"[seasons.dat] no months assigned to season {name}")

        for m in months:
            if m < 1 or m > 12:
                raise ValueError(f"[seasons.dat] invalid month {m} for season {name}")
            if m in month2season:
                raise ValueError(f"[seasons.dat] month {m} assigned to both "
                                 f"'{month2season[m]}' and '{name}'")
            month2season[m] = name

        season_names.append(name)

    missing = [m for m in range(1, 13) if m not in month2season]
    if missing:
        raise ValueError(f"[seasons.dat] months not assigned to any season: {missing}")

    return month2season, season_names

def _images_in(dirpath):
    '''Returns the sorted list of image files contained in a directory'''
    if not os.path.isdir(dirpath):
        return []
    return sorted([f for f in os.listdir(dirpath) if f.lower().endswith(IMG_EXT)])

def _resolve_file(station_name, base, season):
    '''
    Looks for <base>_<season>.dat and falls back to the legacy <base>.dat
    '''
    cand = [os.path.join(station_name, f"{base}_{season}.dat"),
            os.path.join(station_name, f"{base}.dat")]

    for c in cand:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(f"Missing configuration file for season '{season}': "
                            f"expected {cand[0]} (or {cand[1]})")

def _resolve_target(station_name, season):
    '''
    Looks for the target picture of a given season. Accepted layouts:
        images/Target/<season>/<any image>          [recommended]
        images/Target/<something_season>.jpg        [flat, name must contain the season]
        images/Target/<single image>                [legacy, single-target station]
    '''
    tdir = os.path.join(station_name, "images", "Target")

    # 1) dedicated sub-folder
    sub_imgs = _images_in(os.path.join(tdir, season))
    if sub_imgs:
        return os.path.join(tdir, season, sub_imgs[0])

    # 2) flat folder, season name embedded in the file name
    imgs  = _images_in(tdir)
    match = [f for f in imgs
             if season.lower() in os.path.splitext(f)[0].lower()]
    if match:
        return os.path.join(tdir, match[0])

    # 3) legacy single-target station
    if len(imgs) == 1:
        return os.path.join(tdir, imgs[0])

    raise FileNotFoundError(f"No target picture found for season '{season}' in {tdir}")

def load_season_config(station_name, season):
    '''
    Loads everything that is season dependent: settings, GCPs, target picture
    and the camera projection matrix.

    Output
    -----
    cfg : dict with the season configuration
    '''

    filein  = _resolve_file(station_name, "setting", season)
    filecrd = _resolve_file(station_name, "GCP", season)

    (dateINI, dateEND, thr_al, target_col, target_row,
     samP, lon1, lat1, lon2, lat2, ds) = read_set(filein)

    epsg, v, u, x, y, z, filend = read_gcp(filecrd)

    path_target  = _resolve_target(station_name, season)
    image_target = cv2.imread(path_target)
    if image_target is None:
        raise IOError(f"Target picture could not be read: {path_target}")

    # compute camera projection matrix (silenced)
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        A_, A, res = camera_proj(v, u, x, y, z)

    cfg = {"season"      : season,
           "file_set"    : filein,
           "file_gcp"    : filecrd,
           "path_target" : path_target,
           "image_target": image_target,
           "dateINI"     : dateINI,
           "dateEND"     : dateEND,
           "thr_al"      : thr_al,
           "target_col"  : target_col,
           "target_row"  : target_row,
           "samP"        : samP,
           "lon1"        : lon1,
           "lat1"        : lat1,
           "lon2"        : lon2,
           "lat2"        : lat2,
           "ds"          : ds,
           "epsg"        : epsg,
           "xmean"       : np.mean(x),
           "ymean"       : np.mean(y),
           "A"           : A,
           "filend"      : filend}

    print(f"  [{season}] target  : {os.path.relpath(path_target, station_name)}")
    print(f"  [{season}] setting : {os.path.basename(filein)}")
    print(f"  [{season}] GCP     : {os.path.basename(filecrd)} "
          f"(EPSG:{epsg}, sea pixel col/row = {target_col}/{target_row})")

    return cfg

def load_station(station_name):
    '''
    Reads seasons.dat (if present) and loads the configuration of every season.
    If seasons.dat does not exist the station is processed in legacy
    single-target mode (setting.dat + GCP.dat + one picture in images/Target).

    Output
    -----
    month2season : dict {month : season name}
    cfg_all      : dict {season name : season configuration}
    '''

    seasons_file = os.path.join(station_name, "seasons.dat")

    if os.path.isfile(seasons_file):
        month2season, season_names = read_seasons(seasons_file)
        print(f"[Multi-target mode] {len(season_names)} seasons declared: "
              f"{', '.join(season_names)}")
        for s in season_names:
            months = sorted([m for m, ss in month2season.items() if ss == s])
            print(f"  [{s}] months: {months}")
    else:
        month2season = {m: "all" for m in range(1, 13)}
        season_names = ["all"]
        print("[Single-target mode] no seasons.dat found, using setting.dat / GCP.dat")

    cfg_all = {}
    for s in season_names:
        cfg_all[s] = load_season_config(station_name, s)

    # sanity check: the processing window must be the same for every season
    ref = cfg_all[season_names[0]]
    for s in season_names[1:]:
        if (cfg_all[s]["dateINI"] != ref["dateINI"] or
            cfg_all[s]["dateEND"] != ref["dateEND"]):
            print(f"  [WARNING] processing dates of setting_{s}.dat differ from "
                  f"setting_{season_names[0]}.dat -> using the latter")

    return month2season, cfg_all

#%% DATE RECOGNITION

EXIF_DATETIME_ORIGINAL  = 0x9003   # DateTimeOriginal  (when the shot was taken)
EXIF_DATETIME_DIGITIZED = 0x9004   # DateTimeDigitized
EXIF_DATETIME           = 0x0132   # DateTime (last modification, weakest)

EXIF_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y:%m:%d", "%Y-%m-%d")

def _plausible(date):
    '''A picture cannot be older than DATE_MIN nor be taken in the future'''
    if date is None:
        return False
    return DATE_MIN <= date <= datetime.now() + timedelta(days=1)

def _make_date(Y, M, D, h=12, mi=0, s=0):
    '''Builds a datetime and checks that it is a real, plausible date'''
    try:
        date = datetime(int(Y), int(M), int(D), int(h), int(mi), int(s))
    except ValueError:
        return None
    return date if _plausible(date) else None

def exif_date(path_img):
    '''
    Extracts the acquisition date from the EXIF metadata

    Output
    -----
    date : datetime or None
    tag  : name of the EXIF tag used
    '''

    try:
        with Image.open(path_img) as img_PIL:
            exif_data = img_PIL._getexif()
    except Exception:
        return None, None

    if not exif_data:
        return None, None

    tags = ((EXIF_DATETIME_ORIGINAL , "DateTimeOriginal"),
            (EXIF_DATETIME_DIGITIZED, "DateTimeDigitized"),
            (EXIF_DATETIME          , "DateTime"))

    for tag, tagname in tags:
        raw = exif_data.get(tag, None)
        if raw is None:
            continue

        raw = str(raw).strip().strip("\x00").strip()
        for fmt in EXIF_FORMATS:
            try:
                date = datetime.strptime(raw, fmt)
            except ValueError:
                continue
            if _plausible(date):
                return date, tagname

    return None, None

def filename_date(path_img):
    '''
    Guesses the acquisition date from the file name. Recognised patterns:

        20250309              -> 09/03/2025 (12:00)
        2025-03-09, 2025_03_09
        IMG_20250309_142530, PXL_20250309_142530123, 20250309142530
        2025-03-09 14.25.30
        09-03-2025            -> 09/03/2025 (12:00)

    Output
    -----
    date     : datetime or None
    has_time : True if the file name also provided the time of the day
    '''

    stem = os.path.splitext(os.path.basename(path_img))[0]

    # (a) date + time
    pat_dt = (r'(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})'
              r'[-_ tT]{0,3}(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})'
              r'(?:[.,]?\d{1,3})?(?!\d)')
    for m in re.finditer(pat_dt, stem):
        date = _make_date(m.group(1), m.group(2), m.group(3),
                          m.group(4), m.group(5), m.group(6))
        if date is not None:
            return date, True

    # (b) YYYYMMDD only  -> noon is assumed
    pat_d = r'(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)'
    for m in re.finditer(pat_d, stem):
        date = _make_date(m.group(1), m.group(2), m.group(3))
        if date is not None:
            return date, False

    # (c) DDMMYYYY with separators -> noon is assumed
    pat_r = r'(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)'
    for m in re.finditer(pat_r, stem):
        date = _make_date(m.group(3), m.group(2), m.group(1))
        if date is not None:
            return date, False

    return None, False

def download_dates(path_img):
    '''
    Filesystem timestamps of the file. On a downloaded/copied picture these
    reflect the DOWNLOAD date, not the acquisition date.
    '''

    dates = []
    try:
        st = os.stat(path_img)
    except OSError:
        return dates

    stamps = []
    birth = getattr(st, "st_birthtime", None)   # macOS / Windows
    if birth is not None:
        stamps.append(birth)

    if DOWNLOAD_DATE_MODE == "all" or not stamps:
        stamps.append(st.st_ctime)              # inode change (copy/download) on Linux
        stamps.append(st.st_mtime)

    for s in stamps:
        try:
            dates.append(datetime.fromtimestamp(s))
        except (OverflowError, OSError, ValueError):
            pass

    return dates

def get_image_date(path_img, allow_filename=ALLOW_FILENAME_DATE):
    '''
    Acquisition date of a picture, following this priority:

        1. EXIF metadata (DateTimeOriginal > DateTimeDigitized > DateTime)
        2. File name, ONLY if the resulting date does not coincide with the
           download date of the file (otherwise the name most likely encodes
           the moment the picture was downloaded, not the moment it was taken)

    Output
    -----
    date   : datetime or None if no reliable date could be established
    source : short string describing where the date comes from
    '''

    # ---- 1. EXIF -----------------------------------------------------------
    date, tagname = exif_date(path_img)
    if date is not None:
        return date, f"EXIF:{tagname}"

    if not allow_filename:
        return None, "no EXIF date"

    # ---- 2. file name ------------------------------------------------------
    date, has_time = filename_date(path_img)
    if date is None:
        return None, "no EXIF date / no date in file name"

    dls = download_dates(path_img)
    if any(date.date() == d.date() for d in dls):
        return None, f"file-name date ({date.date()}) equals download date"

    return date, "FILENAME" + ("" if has_time else " (time set to 12:00)")

def _move_to_unusable(path_current_img, path_unusable, img):
    '''Moves a picture to the Unusable folder without overwriting'''

    os.makedirs(path_unusable, exist_ok=True)
    dest = os.path.join(path_unusable, img)

    if os.path.exists(dest):
        stem, ext = os.path.splitext(img)
        k = 1
        while os.path.exists(dest):
            dest = os.path.join(path_unusable, f"{stem}_{k}{ext}")
            k += 1
    try:
        os.replace(path_current_img, dest)
    except Exception as e:
        print(f"   Could not move {img} to Unusable: {e}")

def sort_img_by_date(station_name, d1, d2, allow_filename=ALLOW_FILENAME_DATE):
    '''
    Extracts dates for each picture and retains those required by the user

    Input
    -----
    station_name : main directory where pictures are stored
    d1           : Initial date ("0" to process everything)
    d2           : Final date

    Output
    -----
    img_sort  : list of pictures to be processed
    date_sort : DATETIME objects of the pictures to be processed
    src_sort  : where the date of each picture comes from
    '''

    path_shared    = os.path.join(station_name, "images")
    path_image_raw = os.path.join(path_shared, "Raw")
    path_unusable  = os.path.join(path_shared, "Unusable")
    os.makedirs(path_unusable, exist_ok=True)

    # get list of the pictures
    imgs_to_align = sorted([img for img in os.listdir(path_image_raw)
                            if img.lower().endswith(IMG_EXT)])

    records = []
    n_exif, n_name = 0, 0

    for img in imgs_to_align:
        path_current_img = os.path.join(path_image_raw, img)

        try:
            date, source = get_image_date(path_current_img, allow_filename)
        except Exception as e:
            date, source = None, f"error: {e}"

        # no reliable date --> move the picture to the Unusable dir
        if date is None:
            print(f"No reliable date for: {img}  [{source}]")
            print("Moving picture to Unusable folder\n")
            _move_to_unusable(path_current_img, path_unusable, img)
            continue

        if source.startswith("EXIF"):
            n_exif += 1
        else:
            n_name += 1
            print(f"Date of {img} recovered from its file name -> {date} [OK]")

        records.append((img, date, source))

    # select pictures to be processed depending on dates
    process_all = "Y" if d1 == "0" else "N"
    if process_all == "Y":
        date_init = datetime.strptime('19760101', "%Y%m%d")
        date_end  = datetime.strptime('30000101', "%Y%m%d")
        print("\n[Retaining ALL pictures in RAW directory]")
    else:
        date_init = datetime.strptime(d1, "%Y%m%d")
        date_end  = datetime.strptime(d2, "%Y%m%d")
        print("\n[Retaining pictures taken from " + str(date_init) +
              " through " + str(date_end) + "]")

    records = [r for r in records if date_init <= r[1] <= date_end]
    records.sort(key=lambda r: r[1])

    img_sort  = [r[0] for r in records]
    date_sort = [r[1] for r in records]
    src_sort  = [r[2] for r in records]

    print(f"[{len(img_sort)} pictures to process | date from EXIF: {n_exif}, "
          f"from file name: {n_name}]\n")

    return img_sort, date_sort, src_sort

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

    # -------------------- ALTERNATIVES FOR OBJECT DETECTION --------------------
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

    if H is None:
        # homography could not be estimated at all
        return None, 0.0

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

#%% PROJECTION

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

#%% PEOPLE COUNTING SETTINGS
# All the parameters to adapt when moving the people counting to another beach.
# Nothing outside this cell has to be edited for that purpose.

# detector
PERSON_MODEL = "model_yolo/yolov8x.pt"   # weights, path relative to the working directory
PERSON_CONF  = 0.08                      # kept fixed so that counts stay comparable
PERSON_IMGSZ = 2560                      # high: people are small and far away

# counting zone: polygon of the area to EXCLUDE (town, sky and far half of the beach),
# in the aligned frame (4032 x 3024). A person is counted when the CENTRE of its box
# falls OUTSIDE. 
#

POLY_ID = "sturla_near_v2"               # written in the csv, tells which polygon was used
EXCLUDE_POLY = np.array([[2871, 3014], [2249, 1549], [1979, 1533], [1850, 1245],
                         [1627, 1184], [14, 1348], [14, 10], [4031, 19],
                         [4027, 3017]], dtype=np.int32)

# optional per-season override: {"cold": (poly_id, np.array([...])), ...}
# leave empty ({}) to use EXCLUDE_POLY / POLY_ID for all seasons
EXCLUDE_POLY_BY_SEASON = {}

# calibration, measured on 149 photos counted by hand in both zones over 12 months
# rho = n_yolo / real count in the near zone (>1 the detector over-counts on an empty
# beach, <1 it under-counts a crowd) ; k = real WHOLE beach count / n_yolo, a single
# factor merging the detection ratio and the far/near population ratio.
# --> another beach: re-measure both on a manually counted sample 
RHO = {"01": 2.50, "02": 1.15, "03": 1.26, "04": 1.28, "05": 0.98, "06": 0.97,
       "07": 0.80, "08": 0.80, "09": 0.99, "10": 1.31, "11": 1.48, "12": 1.11}
K   = {"01": 3.45, "02": 2.65, "03": 3.35, "04": 4.28, "05": 3.48, "06": 2.48,
       "07": 2.90, "08": 3.32, "09": 2.72, "10": 3.25, "11": 1.49, "12": 2.95}

# output: one csv at the root of the station folder
COUNTS_CSV    = "beach_counts.csv"
COUNTS_FIELDS = ["photo", "date_exif", "month", "season", "n_yolo", "rho_month",
                 "N_near", "k_month", "N_whole", "model", "conf", "imgsz",
                 "poly_id", "processed_on"]


#%% LAUNCH SCRIPT

def process_pic(station_name):
    '''
    Main function aligning every picture on the target of its own season

    Input
    -----
    station_name : main directory of the CoastSnap station
    '''

    # ---------------------------------------------------------------- config
    month2season, cfg_all = load_station(station_name)

    ref = cfg_all[next(iter(cfg_all))]
    dateINI, dateEND = ref["dateINI"], ref["dateEND"]

    # get list of pictures to be processed
    img_sort, date_sort, src_sort = sort_img_by_date(station_name,
                                                     dateINI, dateEND)

    # ----------------------------------------------------------------- paths
    path_shared     = os.path.join(station_name, "images")
    path_image_raw  = os.path.join(path_shared, "Raw")
    path_image_out  = os.path.join(path_shared, "Processed")
    path_unusable   = os.path.join(path_shared, "Unusable")
    path_shape_out  = os.path.join(station_name, "shorelines")

    for p in (path_image_out, path_unusable, path_shape_out):
        os.makedirs(p, exist_ok=True)

    # define SAM model(s) - one predictor per model type, loaded only once
    predictors = {}
    for s, c in cfg_all.items():
        if c["samP"] not in predictors:
            predictors[c["samP"]] = load_predictor(c["samP"])

    # ------------------------------------------------ people counting: setup

    people_model = YOLO(PERSON_MODEL)
    if torch.cuda.is_available():
        person_device = "cuda"
    elif torch.backends.mps.is_available():
        person_device = "mps"
    else:
        person_device = "cpu"
    print(f"[Counting people on {person_device} | {PERSON_MODEL} | polygon {POLY_ID}]\n")

    path_image_bcc = os.path.join(path_shared, "Bcc")
    os.makedirs(path_image_bcc, exist_ok=True)   # imwrite fails silently without it

    csv_path      = os.path.join(station_name, COUNTS_CSV)
    previous_rows = []
    already_done  = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                previous_rows.append(row)
                already_done.add(row["photo"])
        print(f"[{len(already_done)} photos already counted, kept unchanged]\n")

    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=COUNTS_FIELDS,
                                restval="", extrasaction="ignore")
    csv_writer.writeheader()
    for row in previous_rows:
        csv_writer.writerow(row)
    csv_file.flush()
    processed_on = datetime.now().strftime("%Y-%m-%d")

    # ------------------------------------------------------------- main loop
    log_lines = []

    for i in range(len(img_sort)):
        img           = img_sort[i]
        date          = date_sort[i]
        source        = src_sort[i]
        creation_time = date.strftime("%Y_%m_%d-%H_%M_%S")
        date_exif     = date.strftime("%Y-%m-%d %H:%M:%S")  # csv join key
        month         = date.strftime("%m")                 # selects the calibration factor

        # pick the target of the season the picture belongs to
        season = month2season[date.month]
        C      = cfg_all[season]

        print(f"Processing pic {i+1}/{len(img_sort)} | Name: {img} | "
              f"Taken: {creation_time} | Season: {season} | Date from: {source}")

        path_current_img = os.path.join(path_image_raw, img)
        image_out        = os.path.join(path_image_out,
                                        creation_time + "_" + season + ".png")

        status = "OK"

        try:
            # load raw picture and extract matching objects
            image_og = cv2.imread(path_current_img)
            if image_og is None:
                raise IOError("picture could not be read by OpenCV")

            (matches, key_points_images) = find_matches(image_og,
                                                        C["image_target"])

            # Do the homography and the alignement of the image
            image_aligned, algnmt_rate = alignment(image_og, C["image_target"],
                                                   key_points_images, matches)

            if image_aligned is None:
                raise RuntimeError("homography could not be estimated")

            # get back original colors and shape
            image = cv2.cvtColor(image_aligned, cv2.COLOR_BGR2RGB)

            # check if target pixel lies in the border ...
            target_valid = is_black_area(image, C["target_row"], C["target_col"])

            # ... if not, do the segmentation
            if target_valid and algnmt_rate > C["thr_al"]:

                # ------------------------------------- PEOPLE DETECTION SECTION
             
                poly_id_used, poly_used = EXCLUDE_POLY_BY_SEASON.get(
                    season, (POLY_ID, EXCLUDE_POLY))

                det = people_model(image_aligned, imgsz=PERSON_IMGSZ, conf=PERSON_CONF,
                                   classes=[0], device=person_device, verbose=False)[0]
                annotated = image_aligned.copy()   # draw on a copy, never the original
                cv2.polylines(annotated, [poly_used], True, (0, 255, 255), 3)

                n_people = 0
                for box in det.boxes:
                    bx = box.xyxy[0]
                    cx, cy = float((bx[0]+bx[2])/2), float((bx[1]+bx[3])/2)   # box center
                    if cv2.pointPolygonTest(poly_used, (cx, cy), False) >= 0:
                        continue                        # in exclusion zone -> skip
                    x1, y1, x2, y2 = int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])
                    conf_box = float(box.conf[0])       # detection confidence
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{conf_box:.2f}", (x1, max(y1 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                    n_people += 1

                # counter banner (top-right): metadata date + count
                img_h, img_w = annotated.shape[:2]
                text = f"{creation_time}   People: {n_people}"
                font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, img_w / 1600
                thick, margin = max(2, int(font_scale * 2)), int(img_w * 0.01)
                (tw, th), base = cv2.getTextSize(text, font, font_scale, thick)
                tx, ty = img_w - tw - margin, margin + th
                cv2.rectangle(annotated, (tx - margin//2, ty - th - margin//2),
                              (tx + tw + margin//2, ty + base + margin//2), (0, 0, 0), -1)
                cv2.putText(annotated, text, (tx, ty), font, font_scale,
                            (255, 255, 255), thick, cv2.LINE_AA)

                cv2.imwrite(os.path.join(path_image_bcc, creation_time + ".jpg"), annotated)

                # calibrated counts: near zone (de-biased) and whole beach (density numerator)
                if img not in already_done:
                    N_near  = round(n_people / RHO[month], 2)
                    N_whole = round(n_people * K[month], 2)
                    csv_writer.writerow({
                        "photo": img, "date_exif": date_exif, "month": month,
                        "season": season, "n_yolo": n_people, "rho_month": RHO[month],
                        "N_near": N_near, "k_month": K[month], "N_whole": N_whole,
                        "model": os.path.basename(PERSON_MODEL), "conf": PERSON_CONF,
                        "imgsz": PERSON_IMGSZ, "poly_id": poly_id_used,
                        "processed_on": processed_on})
                    csv_file.flush()
                    already_done.add(img)
                # --------------------------------- END PEOPLE DETECTION SECTION

                mask, contours = segment_image(image, predictors[C["samP"]],
                                               C["target_col"], C["target_row"])

                if len(contours) == 0:
                    raise RuntimeError("SAM returned an empty mask")

                # get shoreline pixels
                cnt    = max(contours, key=cv2.contourArea)
                coords = cnt.squeeze()
                vc = coords[:, 0]  # columns
                uc = coords[:, 1]  # rows

                plt.figure()
                plt.imshow(image)
                plt.imshow(mask, alpha=0.4, cmap='gray')
                plt.plot(vc, uc, 'r-', linewidth=1)
                plt.axis('off')
                plt.savefig(image_out, bbox_inches='tight', pad_inches=0, dpi=250)
                plt.close()

                # project shoreline in UTM coord
                world_pts_est = map_utm(C["A"], vc, uc)

                lon = np.add(world_pts_est[:, 0], C["xmean"])
                lat = np.add(world_pts_est[:, 1], C["ymean"])

                # slice coastline to selected area
                in_box = ((lon >= C["lon1"]) & (lon <= C["lon2"]) &
                          (lat >= C["lat1"]) & (lat <= C["lat2"]))
                lon_box, lat_box = lon[in_box], lat[in_box]

                if len(lon_box) < 2:
                    raise RuntimeError("shoreline outside the bounding box")

                # interp to desired resolution
                deltas = np.sqrt(np.diff(lon_box)**2 + np.diff(lat_box)**2)
                s      = np.insert(np.cumsum(deltas), 0, 0)  # distance along line [m]
                s_uniform = np.arange(0, s[-1], C["ds"])

                if len(s_uniform) < 2:
                    raise RuntimeError("shoreline shorter than the interp step")

                lon_intrp = np.interp(s_uniform, s, lon_box)
                lat_intrp = np.interp(s_uniform, s, lat_box)

                # export only interpolated data to shapefile
                geometry = [Point(xy) for xy in zip(lon_intrp, lat_intrp)]

                gdf = gpd.GeoDataFrame(
                    {"type": ["interpolated"] * len(lon_intrp)},
                    geometry=geometry,
                    crs="EPSG:" + str(C["epsg"])
                    )

                shp_name = (creation_time + "_" + season + "_" +
                            C["filend"] + ".shp")
                gdf.to_file(os.path.join(path_shape_out, shp_name))

            else:
                status = (f"alignment/SAM failed (rate={algnmt_rate:.1f}%, "
                          f"thr={C['thr_al']}, target_valid={target_valid})")
                print("Alignment and/or SAM failed")
                print("Moving picture to Unusable folder\n")
                _move_to_unusable(path_current_img, path_unusable, img)

            del image_og
            del image_aligned

        except Exception as e:
            status = f"error: {e}"
            print(f"   {status}")
            print("Moving picture to Unusable folder\n")
            _move_to_unusable(path_current_img, path_unusable, img)

        log_lines.append(f"{img}\t{date}\t{source}\t{season}\t{status}")

    csv_file.close()

    # ------------------------------------------------------------------- log
    filelog = os.path.join(station_name, "processing.log")
    with open(filelog, "a", encoding="utf-8") as f:
        f.write(f"# RUN {datetime.now():%Y-%m-%d %H:%M:%S} | "
                f"{len(img_sort)} pictures\n")
        f.write("# image\tdate\tdate_source\tseason\tstatus\n")
        for line in log_lines:
            f.write(line + "\n")
