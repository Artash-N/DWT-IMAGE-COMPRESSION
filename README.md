# DWT-IMAGE-COMPRESSION


This Notebook contains python code for compression and uncompressing images using Discrete Wavlet Transormations.

It makes use of the pywt library to conduct the wavlet transormations. 

-----------------------------------------------

The "Level 1 Wavlet Compression Demo" and "Level 1 Wavlet Compression Functions.ipynb" notebooks both cover full Wavlet Compression with different thresholds. The "Multilevel Image Compression.ipynb" notebook covers Wavlet Compressions at multiple wavlet levels. (Higher the wavlet level, more accurate the compression, but the longer it takes).

-----------------------------------------------

The DWTCompression.py file contains the actual functions required to compress and uncompress images with Wavlet Compression

Heres an overview of function usage for the 4 functions

1. ```compress_image(image, level, compression_threshold)```

```python
im = imread('rocket.jpg') # Opens sample Image
compressed_wavlets = compress_image(im, level = 1, compression_threshold = 0.60) # Compresses Image into Wavlets and keeping only the top 60% of the wavlet coefficients
```
