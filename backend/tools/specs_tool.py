"""
Tool 2: Tra cứu thông số kỹ thuật (WA12CG5886** & WA14CG5886**) từ trang 50.
"""

class SpecsTool:
    def __init__(self):
        self.specs = {
            "type": "Máy giặt cửa trên (Top Loader)",
            "voltage": "AC 220-240 V / 50 Hz",
            "power_consumption": "1400 - 1650 W",
            "spin_speed": "700 vòng/phút",
            "water_pressure": "0,01 - 0,80 MPa (0,1 - 8,0 kg·f/cm²)",
            "models": {
                "WA12CG5886": {
                    "model_full": "WA12CG5886**",
                    "capacity_kg": 12.0,
                    "dimensions_mm": "R610 x S675 x C1059",
                    "net_weight_kg": 40.0
                },
                "WA14CG5886": {
                    "model_full": "WA14CG5886**",
                    "capacity_kg": 14.0,
                    "dimensions_mm": "R610 x S675 x C1089",
                    "net_weight_kg": 43.0
                }
            },
            "source_page": 50
        }

    def get_specs(self, model_key: str = "all") -> dict:
        model_clean = model_key.upper().strip() if model_key else "ALL"
        if "12" in model_clean:
            return {
                "general": {
                    "type": self.specs["type"],
                    "voltage": self.specs["voltage"],
                    "power_consumption": self.specs["power_consumption"],
                    "spin_speed": self.specs["spin_speed"],
                    "water_pressure": self.specs["water_pressure"],
                },
                "model_details": self.specs["models"]["WA12CG5886"],
                "source_page": self.specs["source_page"]
            }
        elif "14" in model_clean:
            return {
                "general": {
                    "type": self.specs["type"],
                    "voltage": self.specs["voltage"],
                    "power_consumption": self.specs["power_consumption"],
                    "spin_speed": self.specs["spin_speed"],
                    "water_pressure": self.specs["water_pressure"],
                },
                "model_details": self.specs["models"]["WA14CG5886"],
                "source_page": self.specs["source_page"]
            }
        else:
            return self.specs

if __name__ == "__main__":
    tool = SpecsTool()
    print("Specs 14kg:", tool.get_specs("WA14CG"))
