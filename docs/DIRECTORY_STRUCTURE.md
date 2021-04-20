## Python Modules

1.  ### cbbin
    Binarization Module implementing multiple UNet variants

2.  ### cbdiadb
    Visual database Module realising algebra based indexing and mapping between datapoints and the filesystem.
    This module is to a certain extent [PF](https://www.portafontium.eu/) specific. Yet it can easilly be adapted to 
    any other equivalent historical document collection. 

3.  ### cbsegm
    Segmentation module implementing word-proposals given images. Segmentation is performed by morphological 
    manipulation of foreground probability images.  

4. ### cbphocnet
    Word image / string embedding module.

## Programs

1.  ### bin/cb_service
    This is the actual database launching script.

2.  ### bin/cb_cliqbs
    This is a command line web client querying a running database by a string (QBS). 

3.  ### bin/cb_cliqbe
    This is a command line web client querying a running database by an image (QBE/QBR).