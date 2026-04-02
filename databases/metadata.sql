Create Table If Not Exists metadata(
    spID INTEGER PRIMARY KEY,
    spName TEXT NOT NULL,
    retentionIndex TEXT,
    INCHIKEY TEXT NOT NULL,
    formula TEXT NOT NULL,
    mw REAL NOT NULL,
    exactMass INTEGER NOT NULL,
    casNO TEXT NOT NULL,
    comment TEXT,
    numPeaks INTEGER NOT NULL,
    h5ID TEXT NOT NULL,
    h5file TEXT NOT NULL,
    deriv TEXT
)