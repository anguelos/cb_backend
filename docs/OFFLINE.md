### Generate Makefile
```bash
PYTHONPATH="./" bin/cb_create_chronicle_makefile -db_root /data/storage/ -archive_location /data/storage/db3/ -chronicle_paths /data/storage/db3/chronicle/*/*  
```

### Make all documents
```bash
find /data/storage/db3/db_root/   -name \*Makefile -exec  make -f {}   \;  
```
