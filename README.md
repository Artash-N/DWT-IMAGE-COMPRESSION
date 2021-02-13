# DWT-IMAGE-COMPRESSION


This Notebook contains python code for compression and uncompressing images using Discrete Wavlet Transormations.

It makes use of the pywt library to conduct the wavlet transormations. 

-----------------------------------------------

The "Level 1 Wavlet Compression Demo" and "Level 1 Wavlet Compression Functions.ipynb" notebooks both cover full Wavlet Compression with different thresholds. The "Multilevel Image Compression.ipynb" notebook covers Wavlet Compressions at multiple wavlet levels. (Higher the wavlet level, more accurate the compression, but the longer it takes).

-----------------------------------------------

The [DWTCompression](DWTCompression.py) file contains the actual functions required to compress and uncompress images with Wavlet Compression

Heres an overview of usage for the 4 functions in that file:

1. ```compress_image(image, level, compression_threshold)```

```python
im = imread('rocket.jpg') # Opens sample Image

# Setting Wavlets level to 1 (Level can be between 1 and 1000; The higher the level the less the compression loss)
level = 1  


# Compresses Image into Wavlets and keeping only the top 60% of the wavlet coefficients
compressed_wavlets = compress_image(im, level = level, compression_threshold = 0.60) 


# Reconstruct Original Image from Wavlets (with some amount of loss)
# You'll need to provide, in addition to the wavlets, the level used in the compression and the shape of the original image
uncompressed_image = uncompress_image(compressed_wavlets, level, np.shape(im)[:2])


```
