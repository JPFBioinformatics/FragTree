CREATE TABLE IF NOT EXISTS spectra (
    spID            INTEGER PRIMARY KEY,
    spName          TEXT NOT NULL,
    retentionIndex  TEXT,
    INCHIKEY        TEXT NOT NULL,
    formula         TEXT NOT NULL,
    mw              REAL NOT NULL,
    exactMass       REAL NOT NULL,
    casNO           TEXT NOT NULL,
    comment         TEXT,
    numPeaks        INTEGER NOT NULL,
    h5ID            TEXT NOT NULL,
    h5file          TEXT NOT NULL,
    deriv           TEXT
);

CREATE TABLE IF NOT EXISTS molecule (
    molID   INTEGER PRIMARY KEY,
    casNO   TEXT NOT NULL,
    variant INTEGER NOT NULL DEFAULT 0,
    h5ID    TEXT NOT NULL,
    h5file  TEXT NOT NULL,
    UNIQUE (casNO, variant)
);

CREATE TABLE IF NOT EXISTS sp_mol_map (
    spID    INTEGER NOT NULL REFERENCES spectra(spID),
    molID   INTEGER NOT NULL REFERENCES molecule(molID),
    PRIMARY KEY (spID, molID)
);