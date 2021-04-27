### Train PHOCNet

##### Set train and test sets:
```bash
TRAIN_IMAGES=(./data/fake_db/chronicle/archive_name/blovice/*.jp2 ./data/fake_db/chronicle/archive_name/blovice_1/*.jp2 ./data/fake_db/chronicle/archive_name/cheb_gradl/*.jp2 ./data/fake_db/chronicle/archive_name/cheb_gradl_2/*.jp2 ./data/fake_db/chronicle/archive_name/cheb_vyber/*.jp2 ./data/fake_db/chronicle/archive_name/chudenice/*.jp2  ./data/fake_db/chronicle/archive_name/hroznetin/*.jp2 ./data/fake_db/chronicle/archive_name/k_vary/*.jp2 ./data/fake_db/chronicle/archive_name/svojsin/*.jp2 ./data/fake_db/chronicle/archive_name/treben/*.jp2)
TEST_IMAGES=(./data/fake_db/chronicle/archive_name/chudenice_2/*.jp2)
TRAIN_GTS=(./data/fake_db/chronicle/archive_name/blovice/*.gt.json ./data/fake_db/chronicle/archive_name/blovice_1/*.gt.json ./data/fake_db/chronicle/archive_name/cheb_gradl/*.gt.json ./data/fake_db/chronicle/archive_name/cheb_gradl_2/*.gt.json ./data/fake_db/chronicle/archive_name/cheb_vyber/*.gt.json ./data/fake_db/chronicle/archive_name/chudenice/*.gt.json  ./data/fake_db/chronicle/archive_name/hroznetin/*.gt.json ./data/fake_db/chronicle/archive_name/k_vary/*.gt.json ./data/fake_db/chronicle/archive_name/svojsin/*.gt.json ./data/fake_db/chronicle/archive_name/treben/*.gt.json)
TEST_GTS=(./data/fake_db/chronicle/archive_name/chudenice_2/*.gt.json)

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
