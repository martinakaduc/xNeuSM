python make_datasets.py --ds COX2
python make_datasets.py --ds COX2_MD
python make_datasets.py --ds KKI
python make_datasets.py --ds DHFR
python make_datasets.py --ds DBLP-v1
python make_datasets.py --ds MSRC-21

python generate_data_v1.py --config configs/COX2.json
python generate_data_v1.py --config configs/COX2_MD.json
python generate_data_v1.py --config configs/KKI.json
python generate_data_v1.py --config configs/DHFR.json
python generate_data_v1.py --config configs/DBLP-v1.json
python generate_data_v1.py --config configs/MSRC-21.json

python process_data.py COX2 real
python process_data.py COX2_MD real
python process_data.py KKI real
python process_data.py DHFR real
python process_data.py DBLP-v1 real
python process_data.py MSRC-21 real