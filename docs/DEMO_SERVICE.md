###   Documents


## Port 8080
The main intended service. This is where the full indexing will be served.
### Create Indexes
```bash
```

### Serve Indexes
```bash
PYTHONPATH="./" ./bin/cb_service -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt  -indexed_documents /data/storage/new_root/data/all_idx/*pickle -port 8080 -document_root '/data/archiv/public/'
```

## Port 8081
The demo chronicles. These are 10 never seen during any machine learning or design choices. The perceived quality of the indexing on these documents should match the present state capabilities of the system on any chronicle.
### Create Indexes

### Serve Indexes
```bash
PYTHONPATH="./" ./bin/cb_service -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt  -indexed_documents /data/storage/new_root/data/demo_idx/*pickle -port 8081 -document_root '/data/archiv/public/'
```


## Port 8082
### Create Indexes
```bash
DB_ROOT=/data/storage/new_root/data/fake_db_full/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT=/data/storage/new_root/data/fake_db_test_full_docs_resnetidx
MAKEFILE=testfull_phocresnet.mak
mkdir -p $IDX_ROOT
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET
make -f  "$IDX_ROOT/Makefile"
```

### Serve Indexes
```bash
PYTHONPATH="./" ./bin/cb_service -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt  -indexed_documents /data/storage/new_root/data/fake_db_test_full_docs_resnetidx/*pickle -port 8082 -document_root '/data/archiv/public/'
```


## Port 8083
### Create Indexes

### Serve Indexes
```bash
PYTHONPATH="./" ./bin/cb_service -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt  -indexed_documents /data/storage/new_root/data/fake_db_test_resnetidx/*pickle -port 8083 -document_root '/data/archiv/public/'
```



## Evaluation embeddings

### Evaluation pages Groutruthed segmentations
```bash
PHOCNET=/data/storage/new_root/models/phocresnet_0x0.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
MAKEFILE_NAME=test_gtsegm_phocresnet.mak

for DB_ROOT in /data/storage/new_root/fake_db/*/; do  
  IDX_ROOT=/data/storage/new_root/data/fake_db_test_resnetidx_gtsegm/$(basename DB_ROOT)
  mkdir -p $IDX_ROOT
  PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE_NAME" -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .gt.json    -phocnet_path $PHOCNET
  MAKEFILE="${DB_ROOT}"/chronicle/*/*/$MAKEFILE_NAME 
  make -f $MAKEFILE
done

```

### Serve Groutruthed segmentations
```bash
IDX_FILES=/data/storage/new_root/data/fake_db_test_resnetidx_gtsegm
PORT=8083
PYTHONPATH="./" python3  ./bin/cb_service -indexed_documents  "$IDX_ROOT"/*.pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```

### Evaluate Groutruthed segmentations
```bash
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json ./data/fake_db/chudenice_2/chronicle/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold .005 -port 8083
PYTHONPATH="./" python3  ./bin/cb_service -indexed_documents  "$IDX_ROOT"/*.pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```



### Evaluation pages only
```bash
PHOCNET=/data/storage/new_root/models/phocresnet_0x0.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
MAKEFILE_NAME=test_phocresnet.mak

for DB_ROOT in /data/storage/new_root/fake_db/*/; do
  IDX_ROOT=/data/storage/new_root/data/fake_db_test_resnetidx/$(basename DB_ROOT)
  mkdir -p $IDX_ROOT
  PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE_NAME" -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET
  MAKEFILE="${DB_ROOT}"/chronicle/*/*/$MAKEFILE_NAME 
  make -f $MAKEFILE
done

```

