import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from .DatasetReader import DatasetReader
from .Dataunit import DataUnit


class DatasetConvertor:    
    def __init__(self, rawDatasetFolder: Union[str, Path], randomFlag=False):
        self.rawDatasetFolder = Path(rawDatasetFolder) if isinstance(rawDatasetFolder, str) else rawDatasetFolder
        self.dfRaw: Optional[pd.DataFrame] = None
        self.dataUnits: Dict[str, DataUnit] = {}
        self.datasetReader: Optional[DatasetReader] = None
        self.randomFlag = randomFlag
        self._initialize()

    def _initialize(self) -> None:
        self._readRawDataset()
   
    def _readRawDataset(self) -> None:
        self.datasetReader = DatasetReader()
        self.datasetReader.readRawDataset(str(self.rawDatasetFolder), randomFlag=self.randomFlag)
        self.dfRaw = self.datasetReader.dfRaw

    def getDataUnit(self, unitName: str) -> DataUnit:
        if unitName not in self.dataUnits:
            available_units = list(self.dataUnits.keys())
            raise KeyError(
                f"Unit name '{unitName}' not found. "
                f"Available units: {available_units}"
            )
        return self.dataUnits[unitName]

    def addDataUnit(self, config) -> None:
        if self.dfRaw is None or self.dfRaw.empty:
            raise ValueError("Raw dataset has not been loaded. Call _updateRawDataset first.")
        
        self.NAME = config.get("NAME")
        self.LEN_WINDOW = config.get("LEN_WINDOW")
        self.CONTEXT_IDXS = config.get("CONTEXT_IDXS")
        self.DPDR_PARAMS = config.get("DPDR_PARAMS")
        
        dataUnit = DataUnit()
        dataUnit.name = self.NAME
        dataUnit.lenWindow = self.LEN_WINDOW
        dataUnit.timestamps = self.dfRaw.iloc[:, 0].to_numpy()
        dataUnit.setContextData(self.dfRaw.iloc[:, self.CONTEXT_IDXS].to_numpy())
        dataUnit.resampleContextData()
        dataUnit.applyDpDr(
            dbParameter=self.DPDR_PARAMS.get("dbParameter"), 
            alpha=self.DPDR_PARAMS.get("alpha"), 
            mode=self.DPDR_PARAMS.get("mode"))
        self.dataUnits[self.NAME] = dataUnit
   

