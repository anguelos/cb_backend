### Train PHOCNet

##### Set train and test sets:
```bash
DATA_ROOT=./data/fake_db/
TRAIN_CHRONICLES=(blovice_1  blovice_2  cheb_gradl_1  cheb_gradl_2  cheb_vyber  
                  chudenice_1  hroznetin  k_vary  svojsin  treben)
TEST_CHRONICLES=(chudenice_2  plasy)

TRAIN_GTS=()
TRAIN_IMAGES=()
for CHRONICLE in "${TRAIN_CHRONICLES[@]}"; do 
  TRAIN_IMAGES+=($DATA_ROOT/$CHRONICLE/*/*/*/*jp2)
  TRAIN_GTS+=($DATA_ROOT/$CHRONICLE/*/*/*/*.gt.json) 
done  # loop over the array

TEST_GTS=()
TEST_IMAGES=()
for CHRONICLE in "${TEST_CHRONICLES[@]}"; do 
  TEST_IMAGES+=($DATA_ROOT/$CHRONICLE/*/*/*/*jp2)
  TEST_GTS+=($DATA_ROOT/$CHRONICLE/*/*/*/*.gt.json) 
done  # loop over the array
# echo "${TRAIN_IMAGES[@]}" |wc -l

if (( $(echo "${TRAIN_IMAGES[@]}" |wc -w) == 100 )); then
    echo "Train 100 Images OK"
else
    echo "Train Images not OK"
fi 
if (( $(echo "${TEST_IMAGES[@]}" |wc -w) == 20 )); then
    echo "Test 20 Images OK"
else
    echo "Test Images not OK"
fi
if (( $(echo "${TRAIN_GTS[@]}" |wc -w) == 100 )); then
    echo "Train 100 json files OK"
else
    echo "Train json files not OK"
fi 
if (( $(echo "${TEST_GTS[@]}" |wc -w) == 20 )); then
    echo "Test 20 json files OK"
else
    echo "Test json files not OK"
fi

```

##### Train single sample small phocnet:
```bash
PYTHONPATH="./" ./bin/cb_train_phocnet -arch phocnet -batch_size 1 -pseudo_batch_size 10 -train_images "${TRAIN_IMAGES[@]}" -train_gts "${TRAIN_GTS[@]}" -test_images "${TEST_IMAGES[@]}" -test_gts "${TEST_GTS[@]}"  -epochs 200

```

##### Train single sample small phoc-Resnet:
```bash
PYTHONPATH="./" ./bin/cb_train_phocnet -arch phocresnet -batch_size 1 -pseudo_batch_size 10 -train_images "${TRAIN_IMAGES[@]}" -train_gts "${TRAIN_GTS[@]}" -test_images "${TEST_IMAGES[@]}" -test_gts "${TEST_GTS[@]}"  -epochs 200

```

##### Train single sample large phoc-Resnet:
```bash
PYTHONPATH="./" ./bin/cb_train_phocnet -arch phocresnet -batch_size 1 -pseudo_batch_size 10 -train_images "${TRAIN_IMAGES[@]}" -train_gts "${TRAIN_GTS[@]}" -test_images "${TEST_IMAGES[@]}" -test_gts "${TEST_GTS[@]}"  -epochs 200  -pyramid_levels 1,2,3,4,5,7,11 -resume_path ./models/phocresnet_p1_2_3_4_5_7_11_0x0.pt

```
