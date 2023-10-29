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

python process_data.py --data_name COX2 --real
python process_data.py --data_name COX2_MD --real
python process_data.py --data_name KKI --real
python process_data.py --data_name DHFR --real
python process_data.py --data_name DBLP-v1 --real
python process_data.py --data_name MSRC-21 --real