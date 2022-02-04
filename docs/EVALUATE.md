## Create Databases 

### Chudenice
```bash
export PYTHONPATH="./:./thirdparty/iunets/"

DOCNAME=chronicle/soap-kt/soap-kt_00780_mesto-chudenice-1924-1936
DB_ROOT=./data/fake_db/chudenice_2/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0_v2.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT_REAL_SEGM=./data/evaluation/real_segm/
IDX_ROOT_GT_SEGM=./data/evaluation/gt_segm/
MAKEFILE=test_phocresnetv2_real_segmentation_chudenice_2.mak

mkdir -p "$IDX_ROOT_REAL_SEGM/"   "$IDX_ROOT_GT_SEGM/"
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT_REAL_SEGM -chronicle_paths "$DB_ROOT/$DOCNAME/" -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET    

make -f "$DB_ROOT/$DOCNAME/$MAKEFILE"
./bin/cb_embed_proposals -phocnet $PHOCNET -docname $DOCNAME -db_root $DB_ROOT -words_files "$DB_ROOT/$DOCNAME/"*.gt.json  -image_files "$DB_ROOT/$DOCNAME/"*.jp2 -output_basename "$IDX_ROOT_GT_SEGM/"

```

### Plasy
```bash
export PYTHONPATH="./:./thirdparty/iunets/"

DOCNAME=chronicle/soap-ps/soap-ps_00129_mesto-plasy-1958-1965
DB_ROOT=./data/fake_db/plasy/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0_v2.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT_REAL_SEGM=./data/evaluation/real_segm/
IDX_ROOT_GT_SEGM=./data/evaluation/gt_segm/
MAKEFILE=test_phocresnetv2_real_segmentation_plasy.mak

mkdir -p "$IDX_ROOT_REAL_SEGM/"   "$IDX_ROOT_GT_SEGM/"
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT_REAL_SEGM -chronicle_paths "$DB_ROOT/$DOCNAME/" -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET    

make -f "$DB_ROOT/$DOCNAME/$MAKEFILE"
./bin/cb_embed_proposals -phocnet $PHOCNET -docname $DOCNAME -db_root $DB_ROOT -words_files "$DB_ROOT/$DOCNAME/"*.gt.json  -image_files "$DB_ROOT/$DOCNAME/"*.jp2 -output_basename "$IDX_ROOT_GT_SEGM/"

```

### Svojin
```bash
export PYTHONPATH="./:./thirdparty/iunets/"

DOCNAME=chronicle/soap-tc/soap-tc_00641_dekanstvi-svojsin-1840-1973
DB_ROOT=./data/fake_db/svojsin_2/
PHOCNET=/data/storage/new_root/models/phocresnet_0x0_v2.pt
BINNET=/data/storage/new_root/models/srunet.pt
BOXNET=/data/storage/new_root/models/box_iou.pt
IDX_ROOT_REAL_SEGM=./data/evaluation/real_segm/
IDX_ROOT_GT_SEGM=./data/evaluation/gt_segm/
MAKEFILE=test_phocresnetv2_real_segmentation_svojsin_2.mak

mkdir -p "$IDX_ROOT_REAL_SEGM/"   "$IDX_ROOT_GT_SEGM/"
PYTHONPATH="./" bin/cb_create_chronicle_makefile -make_filename "$MAKEFILE" -db_root $DB_ROOT  -archive_location $IDX_ROOT_REAL_SEGM -chronicle_paths "$DB_ROOT/$DOCNAME/" -aux_proposals_postfix .words.json    -phocnet_path $PHOCNET    

make -f "$DB_ROOT/$DOCNAME/$MAKEFILE"
./bin/cb_embed_proposals -phocnet $PHOCNET -docname $DOCNAME -db_root $DB_ROOT -words_files "$DB_ROOT/$DOCNAME/"*.gt.json  -image_files "$DB_ROOT/$DOCNAME/"*.jp2 -output_basename "$IDX_ROOT_GT_SEGM/"

```


### Evaluate Proposals
```bash
CHRONICLE="chudenice_2"
echo $CHRONICLE
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_evaluate_proposals -proposals ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*words.json -gt ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*.gt.json
echo

CHRONICLE="plasy"
echo $CHRONICLE
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_evaluate_proposals -proposals ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*words.json -gt ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*.gt.json
echo

CHRONICLE="svojsin_2"
echo $CHRONICLE
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_evaluate_proposals -proposals ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*words.json -gt ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*.gt.json
echo

CHRONICLE="all"
echo $CHRONICLE
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_evaluate_proposals -proposals ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*words.json -gt ./data/fake_db/"$CHRONICLE"/chronicle/*/*/*.gt.json
echo


```



### Launch Services 

Five services in background one service in foreground. In order kill the services simply run in a new shell kill the terminal when done.

```bash
PORT=8084
IDX=./data/evaluation/real_segm/soap-kt_00780_mesto-chudenice-1924-1936.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0_v2.pt -port $PORT &


PORT=8085
IDX=./data/evaluation/real_segm/soap-ps_00129_mesto-plasy-1958-1965.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT &


PORT=8086
IDX=./data/evaluation/real_segm/soap-tc_00641_dekanstvi-svojsin-1840-1973.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT &


PORT=8087
IDX=./data/evaluation/gt_segm/soap-kt_00780_mesto-chudenice-1924-1936.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0_v2.pt -port $PORT &


PORT=8088
IDX=./data/evaluation/gt_segm/soap-ps_00129_mesto-plasy-1958-1965.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT &


PORT=8089
IDX=./data/evaluation/gt_segm/soap-tc_00641_dekanstvi-svojsin-1840-1973.pickle
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX  -embeding_net /data/storage/new_root/models/phocresnet_0x0.pt -port $PORT &


```

#### Kill evaluation servers
```bash
kill -9 $(lsof -i :8084|tail -n 1|awk '{print $2}')
kill -9 $(lsof -i :8085|tail -n 1|awk '{print $2}')
kill -9 $(lsof -i :8086|tail -n 1|awk '{print $2}')
kill -9 $(lsof -i :8087|tail -n 1|awk '{print $2}')
kill -9 $(lsof -i :8088|tail -n 1|awk '{print $2}')
kill -9 $(lsof -i :8089|tail -n 1|awk '{print $2}')

```


### Evaluate Service

#### Real Segmentation
```bash
IOU_THRESHOLD=.01
EVAL_ROOT="./data/evaluation/real_segm/"

CHRONICLE=chudenice_2
PORT=8084
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT | tee "$EVAL_ROOT/$CHRONICLE.log" 


CHRONICLE=plasy
PORT=8085
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT | tee "$EVAL_ROOT/$CHRONICLE.log" 

CHRONICLE=svojsin_2
PORT=8086
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT | tee "$EVAL_ROOT/$CHRONICLE.log" 


```

#### Groundtruth Segmentation
```bash
IOU_THRESHOLD=.01
EVAL_ROOT="./data/evaluation/gt_segm/"

CHRONICLE=chudenice_2
PORT=8087
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT -save_data_path "$EVAL_ROOT/evaluation_dump_$CHRONICLE.pickle" | tee "$EVAL_ROOT/$CHRONICLE.log" 


IOU_THRESHOLD=.01
EVAL_ROOT="./data/evaluation/gt_segm/"
CHRONICLE=plasy
PORT=8088
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT -save_data_path "$EVAL_ROOT/evaluation_dump_$CHRONICLE.pickle" | tee "$EVAL_ROOT/$CHRONICLE.log"

IOU_THRESHOLD=.01
EVAL_ROOT="./data/evaluation/gt_segm/"
CHRONICLE=svojsin_2
PORT=8089
PYTHONPATH="./" ./bin/cb_evaluate_service -gt_json "./data/fake_db/$CHRONICLE/chronicle"/*/*/*gt.json -rectangles_per_document 10000000 -iou_threshold $IOU_THRESHOLD -port $PORT -save_data_path "$EVAL_ROOT/evaluation_dump_$CHRONICLE.pickle" | tee "$EVAL_ROOT/$CHRONICLE.log"

```
