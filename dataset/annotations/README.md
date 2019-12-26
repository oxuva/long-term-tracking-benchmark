The file `dev.csv` follows the OxUvA format and contains annotations for the OxUvA long-term tracking _dev_ set (which is constructed using videos from the YTBB _validation_ set).
The file `dev_constrained_ytbb_train.csv` follows the YTBB format and contains annotations for the subset of the YTBB _train_ set which can be used for development in the "constrained" track.
Both files only contain annotations for the dev classes.
Whereas the tracks in `dev.csv` are comprised of multiple 20-second annotation segments, the tracks in `dev_ytbb_train_constrained.csv` are the original 20-second segments from YTBB.
