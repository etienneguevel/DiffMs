# build datadir file structure
mkdir -p data/
mkdir -p data/fp2mol/
mkdir -p data/fp2mol/raw/

cd data/fp2mol/raw/

# download raw data
curl https://hmdb.ca/system/downloads/current/structures.zip -o structures.zip
unzip structures.zip

curl https://clowder.edap-cluster.com/api/files/6616d8d7e4b063812d70fc95/blob -o blob
unzip blob

curl https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip -o coconut_csv-03-2025.zip
unzip coconut_csv-03-2025.zip

curl https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv -o dataset_v1.csv
mv dataset_v1.csv moses.csv
