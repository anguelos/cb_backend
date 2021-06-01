## Gt Segmentation

### Extract with gt
```bash
DB_ROOT=/data/storage/new_root/fake_db/chudenice_2/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT=/data/storage/new_root/data/fake_db_test_resnetidx_gtseg
MAKEFILE=test_phocresnet_gt_segmentation.mak
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .gt.json    -phocnet_path $PHOCNET    

make -f /data/storage/new_root/fake_db/chudenice_2/chronicle/soap-kt/soap-kt_00780_mesto-chudenice-1924-1936/test_phocresnet_gt_segmentation.mak
```


### Launch Service
```bash
PORT=8082
PYTHONPATH="./" ./bin/cb_service -indexed_documents  /data/storage/new_root/data/fake_db_test_resnetidx_gtseg/soap-kt_00780_mesto-chudenice-1924-1936.pickle  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT

```

### Evaluate Service
```bash
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json ./data/fake_db/chudenice_2/chronicle/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold .005 -port 8082
```

## Real Segmentation

### Extract with gt
```bash
DB_ROOT=/data/storage/new_root/fake_db/chudenice_2/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT=/data/storage/new_root/data/fake_db_test_resnetidx
MAKEFILE=test_phocresnet_gt_segmentation.mak
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET    

make -f /data/storage/new_root/fake_db/chudenice_2/chronicle/soap-kt/soap-kt_00780_mesto-chudenice-1924-1936/test_phocresnet.mak
```


### Launch Service
```bash
PORT=8082
PYTHONPATH="./" ./bin/cb_service -indexed_documents  /data/storage/new_root/data/fake_db_test_resnetidx/soap-kt_00780_mesto-chudenice-1924-1936.pickle  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT

```

### Evaluate Service
```bash
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json ./data/fake_db/chudenice_2/chronicle/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold .005 -port 8082
```