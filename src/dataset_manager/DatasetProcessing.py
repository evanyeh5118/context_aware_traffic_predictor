import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

from .DatasetReader import DatasetReader
from .Dataunit import DataUnit


class DatasetConvertor:    
    # Constants for finger names
    FINGER_NAMES = ['thumb', 'index', 'middle']
    DIRECTION_MAPPING = {"forward": "fr", "backward": "bk"}
    
    # Default context indices
    DEFAULT_CONTEXT_FORWARD = {
        "thumb": [1, 2, 3],
        "index": [5, 6, 7],
        "middle": [9, 10, 11],
    }
    DEFAULT_CONTEXT_BACKWARD = {
        "thumb": [4],
        "index": [8],
        "middle": [12],
    }
    
    def __init__(self, rawDatasetFolder: Union[str, Path]):
        self.rawDatasetFolder = Path(rawDatasetFolder) if isinstance(rawDatasetFolder, str) else rawDatasetFolder
        self.dfRaw: Optional[pd.DataFrame] = None
        self.fingerDataUnits: Dict[str, DataUnit] = {}
        self.idxsContext: Dict[str, Dict[str, List[int]]] = {}
        self.datasetReader: Optional[DatasetReader] = None
        self._initialize()

    def _initialize(self) -> None:
        self._configuration()
        self._updateRawDataset(self.rawDatasetFolder)
        self._separateFingersDataByDirections()

    def _configuration(
            self, 
            idxsContext: Optional[Dict[str, Dict[str, List[int]]]] = None
        ) -> None:
        self.idxsContext = idxsContext if idxsContext is not None else {
            "forward": self.DEFAULT_CONTEXT_FORWARD,
            "backward": self.DEFAULT_CONTEXT_BACKWARD
        }
        
    def _separateFingersDataByDirections(self) -> None:
        if self.dfRaw is None or self.dfRaw.empty:
            raise ValueError("Raw dataset has not been loaded. Call _updateRawDataset first.")
        
        self.fingerDataUnits = {}
        for direction in self.DIRECTION_MAPPING.keys():
            for fingerName, idxsContext in self.idxsContext[direction].items():
                dataUnit = DataUnit()
                dataUnit.name = fingerName
                dataUnit.setContextData(self.dfRaw.iloc[:, idxsContext].to_numpy())
                dataUnit.timestamps = self.dfRaw.iloc[:, 0].to_numpy()
                self.fingerDataUnits[f"{fingerName}_{self.DIRECTION_MAPPING[direction]}"] = dataUnit

    def _updateRawDataset(self, rawDatasetFolder: Union[str, Path]) -> None:

        self.datasetReader = DatasetReader()
        self.rawDatasetFolder = Path(rawDatasetFolder) if isinstance(rawDatasetFolder, str) else rawDatasetFolder
        self.datasetReader.readRawDataset(str(self.rawDatasetFolder))
        self.dfRaw = self.datasetReader.dfRaw

    def getDataUnit(self, unitName: str) -> DataUnit:

        if unitName not in self.fingerDataUnits:
            available_units = list(self.fingerDataUnits.keys())
            raise KeyError(
                f"Unit name '{unitName}' not found. "
                f"Available units: {available_units}"
            )
        return self.fingerDataUnits[unitName]
    
    def processDataset(
            self, 
            direction: str, 
            dbParameter: float = 0.01, 
            alpha: float = 0.01, 
            mode: str = "fixed", 
            verbose: bool = True
        ) -> Dict[str, float]:

        if direction not in self.DIRECTION_MAPPING:
            raise ValueError(
                f"Invalid direction '{direction}'. "
                f"Must be one of: {list(self.DIRECTION_MAPPING.keys())}"
            )
        
        direction_suffix = self.DIRECTION_MAPPING[direction]
        compression_rates = {}
        
        for fingerName in self.FINGER_NAMES:
            unitName = f"{fingerName}_{direction_suffix}"
            
            if unitName not in self.fingerDataUnits:
                if verbose:
                    print(f"Warning: Unit '{unitName}' not found, skipping...")
                continue
            
            if verbose:
                print(f"========== {fingerName.capitalize()} ============")
            
            dataUnit = self.fingerDataUnits[unitName]
            dataUnit.resampleContextData()
            dataUnit.applyDpDr(dbParameter=dbParameter, alpha=alpha, mode=mode)
            dataUnit.interpolateCotextAfterDpDr()
            
            compression_rates[fingerName] = dataUnit.compressionRate
            
            if verbose:
                print(f"{direction.capitalize()}: Compression rate: {compression_rates[fingerName]:.4f}")
        
        return compression_rates
    '''
    def saveDataset(self, folder: Union[str, Path]) -> None:
        folder = Path(folder) if isinstance(folder, str) else folder
        folder.mkdir(parents=True, exist_ok=True)
        for fingerName in self.FINGER_NAMES:
            for direction in self.DIRECTION_MAPPING.keys():
                unitName = f"{fingerName}_{self.DIRECTION_MAPPING[direction]}"
                self.fingerDataUnits[unitName].save(folder) 
    '''