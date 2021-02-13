# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:37:50 2021

@author: Artash Nath

Co-Founder, HotPopRobot.com
Twitter: @wonrobot
"""


"""
This Python File contains neccesary functions to apply different
levels of Wavlet compression to an Image, and uncompress it while
keeping different percentages of Wavlet thresholds
"""

# Importing Neccesary Libaries

import pywt # For wavlets
import numpy # Handling Array Data
import numpy as np # ^
from PIL import Image # Image Operations
import cv2 # More Image Operations
import matplotlib.pyplot as plt # Plots and displays
from skimage.io import imread # Even more image operations

import warnings
warnings.filterwarnings("ignore")
##############################--HELPER-FUNCTIONS--########################################
# Calculate pixel simmilarity between 2 images of same size

def calculate_image_simmilarity(imageA, imageB, dontprint=False):
    imageA = imageA.astype(np.float32) # Neccesary for the substraction to work
    imageB = imageB.astype(np.float32)
    diff = abs(imageA - imageB) # Find all pixel differences between 2 images
    err = diff.mean() # Calculate Average Difference between those 2 images
    if not dontprint:
        print("Mean Pixel Error between original and reconstructed image is : {}".format(err))
    return err

    
    
# this function returns a threshold, where all pixels below that threshold should be discareded
# based on the percentage of pixels we want to discard (compression_level)
# For example, if we wanted to keep only (50%) of the values in the image, we would run
# "im, 0.5" into the function, and it would return the "threshold value"
# where 50% of the image's values are below this threshold
def get_threshold(im, compression_level):
    thresh = np.sort(abs((im.ravel())))[int(im.size*(1-compression_level))]
    return thresh


# Generates Slice "Instructions" for reconstruction based on original image size
def generate_wavlet_slices(x_shape, y_shape, level):
    sample_image = np.random.normal(size=(x_shape, y_shape))
    sample_coeff = pywt.wavedec2(sample_image, 'db2', mode='periodization', level=level)
    slices_instructions = pywt.coeffs_to_array(sample_coeff)[1]
    return slices_instructions

#########################--RGB-COMPRESSION-FUNCTIONS--####################################

# Compresses RGB image into Wavlet Thresholds based on
# Image, Wavlet Level, and the compression_threshold (from 0 - 1, what threshold of coefficients do you want to keep?)
# (1 is near-losless compression as your keeping all the coefficients)
def compress_image(image, level, compression_threshold):
    
    # Breaking Image into RGB Layers (Wavlet Compression works on 2D arrays only)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    
    # Compressing each layer into it's wavlet coefficients
    # With the level defined in the function
    r_coeff = pywt.wavedec2(r, 'db2', mode='periodization', level=level)
    g_coeff = pywt.wavedec2(g, 'db2', mode='periodization', level=level)
    b_coeff = pywt.wavedec2(b, 'db2', mode='periodization', level=level)
    
    # Breaking each Coefficient into it's coefficient array, and a metadata variable (doesnt change from image to image, only changed based on original size)
    arr_r = pywt.coeffs_to_array(r_coeff)[0]
    arr_g = pywt.coeffs_to_array(g_coeff)[0]
    arr_b = pywt.coeffs_to_array(b_coeff)[0]
    
    # Getting actual value threshold (below what value of pixel to keep in image) based on percent_threshold provided above
    threshold_r = get_threshold(arr_r, compression_threshold)
    threshold_g = get_threshold(arr_g, compression_threshold)
    threshold_b = get_threshold(arr_b, compression_threshold)
    
    # Compressing each coefficient array based on calculated value threshold 
    arr_r_compressed = arr_r * (abs(arr_r) > (threshold_r))
    arr_g_compressed = arr_g * (abs(arr_g) > (threshold_g))
    arr_b_compressed = arr_b * (abs(arr_b) > (threshold_b)) 
    
    compressed_image = (np.stack((arr_r_compressed, arr_g_compressed, arr_b_compressed), 2))
    
    return compressed_image

    
# Uncompressed wavlet into reconstructed image based on
# Wavlets, Wavlet Level used to compress, and shape of ORIGINAL image
def uncompress_image(compressed_image, level, im_shape):
    
    # Breaking Compressed Wavlet of Image into RGB Layers (Wavlet Compression works on 2D arrays only)
    arr_r_compressed = compressed_image[:,:,0]
    arr_g_compressed = compressed_image[:,:,1]
    arr_b_compressed = compressed_image[:,:,2]
    
    x_shape, y_shape = im_shape
    slices = generate_wavlet_slices(x_shape, y_shape, level=level)

    # Converting Compressed Coefficients back into Original Wavlet form (which the pywt library can work with)    
    r_coeff = pywt.array_to_coeffs(arr_r_compressed, slices, output_format='wavedecn')
    g_coeff = pywt.array_to_coeffs(arr_g_compressed, slices, output_format='wavedecn')
    b_coeff = pywt.array_to_coeffs(arr_b_compressed, slices, output_format='wavedecn')
    
    # Using PYWT to uncompress the wavlet coefficients back into their image form
    r_reconstructed = pywt.waverecn(r_coeff, 'db2', mode='periodization').astype(np.uint8)
    g_reconstructed = pywt.waverecn(g_coeff, 'db2', mode='periodization').astype(np.uint8)
    b_reconstructed = pywt.waverecn(b_coeff, 'db2', mode='periodization').astype(np.uint8)
    
    
    # Stacking the uncompressed R. G. B. images into the final RGB image
    reconstructed_im = (np.stack((r_reconstructed, g_reconstructed, b_reconstructed), 2)).astype(np.uint8)

    return reconstructed_im

#########################--COMPRESSION_ANALYSIS_GRAPHS--####################################

# Graph the "wavlet level" of a compression versus the Mean(Original - Recreated) (loss between original and uncompressed image)
# Input Parameters:
# Image to graph for
# level_min, level_max (level Range to graph for)
# Compression_Threshold (what threshold of wavlet coefficients to KEEP during compression)
# Resolution (number of sample levels to pick between level_min and level_max)
def wavlet_level_graph(im, level_min, level_max, compression_threshold, resolution=20):
    
    if (level_min <1) or (level_min >999):
        raise ValueError("Level_Min must be between 1 and 999")
    if (level_max <2) or (level_max >1000):
        raise ValueError("Level_Max must be between 2 and 1000")
    if (level_min == level_max):
        raise ValueError("Level_Min and Level_Max must be different")
    if (level_max < level_min):
        raise ValueError("Level_Max must be greater then Level_Min")
        
    x_shape, y_shape = np.shape(im)[:2]
    test_levels = (np.linspace(level_min, level_max, resolution)).astype(np.uint16)

    test_level_simmilarities = []
    
    for level in test_levels:
        ci = compress_image(im, level, compression_threshold)
        ui = uncompress_image(ci, level, (x_shape, y_shape))
        test_level_simmilarities.append(calculate_image_simmilarity(im, ui, dontprint=True))
        
    plt.figure(figsize=(15,7))
    plt.title("Wavlet Compression Level versus Image Simmilarity ({}% Compression)".format(compression_threshold), size=18)
    plt.xlabel("Level", size = 15)
    plt.ylabel("Mean(Original - Recreated)", size = 15)
    plt.plot(test_levels, test_level_simmilarities)
    
    
    
# Graph the "wavlet coefficient threshold" of a compression versus the Mean(Original - Recreated) (loss between original and uncompressed image)
# Input Parameters:
# Image to graph for
# threshold_min, threshold_max (Threshold Range to graph for)
# level (Wavlet Level to use for Compression)
# Resolution (number of sample thresholds to pick between threshold_min and threshold_max)
def wavlet_threshold_graph(im, level, min_threshold=1e-3, max_threshold=1, resolution = 20):
    
    if (level<1) or (level>1000):
        raise ValueError("Level must be between 1 and 1000")
    if (min_threshold<1e-3) or (min_threshold>1):
        raise ValueError("Level must be between 1e-3 and 1")
        
    x_shape, y_shape = np.shape(im)[:2]
    test_thresholds = (np.linspace(min_threshold, max_threshold, resolution))
    simmilarities = []
    
    for threshold in test_thresholds:
        ci = compress_image(im, level, threshold)
        ui = uncompress_image(ci, level, (x_shape, y_shape))
        simmilarities.append(calculate_image_simmilarity(im, ui, dontprint=True))
        
        
    plt.figure(figsize=(12,6))
    
    plt.plot(test_thresholds, simmilarities)
    
    plt.xlabel("Kept DCT Thresholds (%)", size = 15)
    plt.ylabel("Mean(Original - Recreated)", size = 15)
    plt.title("Reconstructed Image Quality on LOG Scale (Level {})".format(level), size = 18)
    plt.yscale('log')

################################################################################################
