"""
Utility script to convert pickle config files to JSON format.
This handles the module path issue by using custom unpickler.
"""

import pickle
import sys
import json
from dataclasses import asdict

# Add custom unpickler to handle old module paths
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        """Remap old module paths to new ones."""
        # Map old context_free module to current structure
        if 'context_free' in module:
            module = module.replace('context_free.', '')
            if module.startswith('src.'):
                module = module[4:]  # Remove 'src.' prefix if present
        
        # Try current module structure
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Try src.config for config classes
            if name in ['MetaConfig', 'ModelConfig', 'TrainingConfig']:
                try:
                    return super().find_class('src.config', name)
                except:
                    pass
            
            # If still failing, try to recreate the class dynamically
            print(f"Warning: Could not find {module}.{name}, attempting manual extraction...")
            raise

def convert_pickle_to_json(pickle_path, json_path):
    """Convert a pickle file to JSON."""
    try:
        with open(pickle_path, 'rb') as f:
            obj = CustomUnpickler(f).load()
        
        # Convert to dict if it's a dataclass
        if hasattr(obj, '__dataclass_fields__'):
            data = asdict(obj)
        elif isinstance(obj, dict):
            data = obj
        else:
            # For other objects, try to extract __dict__
            data = obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Successfully converted {pickle_path} -> {json_path}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Failed to convert {pickle_path}: {e}")
        return False

def main():
    import os
    
    model_folder = "model"
    config_name = "combined_flows_forward"
    
    # Files to convert
    conversions = [
        (f"{model_folder}/{config_name}_modelConfig.pkl", 
         f"{model_folder}/{config_name}_modelConfig.json"),
        (f"{model_folder}/{config_name}_metaConfig.pkl", 
         f"{model_folder}/{config_name}_metaConfig.json"),
    ]
    
    print("Starting pickle to JSON conversion...")
    print("=" * 60)
    
    success_count = 0
    for pickle_path, json_path in conversions:
        if os.path.exists(pickle_path):
            if convert_pickle_to_json(pickle_path, json_path):
                success_count += 1
        else:
            print(f"✗ File not found: {pickle_path}")
    
    print("=" * 60)
    print(f"Conversion complete: {success_count}/{len(conversions)} successful")
    
    if success_count == len(conversions):
        print("\n✓ All files converted successfully!")
        print("You can now use .json files instead of .pkl files")
    else:
        print("\n⚠ Some conversions failed. You may need to regenerate those configs.")

if __name__ == "__main__":
    main()

