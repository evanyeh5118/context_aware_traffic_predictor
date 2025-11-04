import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from .DatasetReader import DatasetReader
from .Dataunit import DataUnit


class DatasetConvertor:    
    def __init__(self, rawDatasetFolder: Union[str, Path], configs):
        self.rawDatasetFolder = Path(rawDatasetFolder) if isinstance(rawDatasetFolder, str) else rawDatasetFolder
        self.dfRaw: Optional[pd.DataFrame] = None
        self.fingerDataUnits: Dict[str, DataUnit] = {}
        self.idxsContext: Dict[str, Dict[str, List[int]]] = {}
        self.datasetReader: Optional[DatasetReader] = None
        
        self.configs = configs
        self.FINGER_NAMES = list(configs.get("FINGER_NAMES", []))
        self.DIRECTION_MAPPING = dict(configs.get("DIRECTION_MAPPING", {}))
        self.DEFAULT_CONTEXT_FORWARD = dict(configs.get("DEFAULT_CONTEXT_FORWARD", {}))
        self.DEFAULT_CONTEXT_BACKWARD = dict(configs.get("DEFAULT_CONTEXT_BACKWARD", {}))

        self._initialize()

    def _initialize(self) -> None:
        self._load_config_from_file()
        self._configuration()
        self._updateRawDataset(self.rawDatasetFolder)
        self._separateFingersDataByDirections()

    def _load_config_from_file(self) -> None:
        """Load configuration from experiments/config/dataset_convertor_config.json if available."""
        try:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "experiments" / "config" / "dataset_convertor_config.json"
            if not config_path.exists():
                return

            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)

            if isinstance(cfg.get("FINGER_NAMES"), list):
                self.FINGER_NAMES = cfg["FINGER_NAMES"]

            if isinstance(cfg.get("DIRECTION_MAPPING"), dict):
                self.DIRECTION_MAPPING = cfg["DIRECTION_MAPPING"]

            if isinstance(cfg.get("DEFAULT_CONTEXT_FORWARD"), dict):
                self.DEFAULT_CONTEXT_FORWARD = cfg["DEFAULT_CONTEXT_FORWARD"]

            if isinstance(cfg.get("DEFAULT_CONTEXT_BACKWARD"), dict):
                self.DEFAULT_CONTEXT_BACKWARD = cfg["DEFAULT_CONTEXT_BACKWARD"]
        except Exception:
            # Silently fall back to defaults if any issue occurs while loading config
            pass

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
                dataUnit.timestamps = self.dfRaw.iloc[:, 0].to_numpy()
                dataUnit.setContextData(self.dfRaw.iloc[:, idxsContext].to_numpy())
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
            
            compression_rates[fingerName] = dataUnit.compressionRate
            
            if verbose:
                print(f"{direction.capitalize()}: Compression rate: {compression_rates[fingerName]:.4f}")
        
        return compression_rates
    
