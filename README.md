# HMRLP
These are codes for the routineness model and its corresponding data of the paper "Unraveling Human Mobility Routineness: Implications for Next Location Prediction", plus a subset of the anonymized and processed data for the prediction models containing 10.000 users.

### Data
In the "data" folder:
* dataset.tar.gz — Data for the routineness model, including 151 .parquet files, each of them containing 1,000 anonymized users' hourly behavioral observations.
* subset_train.csv — A subset of the mobility data for training on prediction models, containing 10,000 anonymized users.
* subset_test.csv — A subset of the mobility data for testing, containing the same 10,000 anonymized users.
### Code
* routineness.stan — A Stan program, that allowed us to measure the weight of the routine/random behaviors of each individual and to calculate their routineness.
* run.py — A Python file to run the Stan program. Noted that a package, CmdStanPy, must be installed. Under the hood, CmdStanPy uses the CmdStan command line interface to compile and run a Stan program.
