from pathlib import Path
import importlib.util
import sys
import numpy as np

path = Path("src/evaluation/faithful_explanation.py")
spec = importlib.util.spec_from_file_location("signalscope_faithful", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

heat = np.zeros((32, 32), dtype=np.float32)
heat[8:16, 10:18] = 1.0

loc = module.analyze_localization(heat)
assert loc.localized
assert loc.bbox is not None

explanation = module.build_grounded_evidence(
    ai_probability=0.91,
    predicted_class=1,
    localization=loc,
)
joined = " ".join(explanation["evidence"]).lower()
assert "fingers" not in joined
assert "reflection is physically impossible" not in joined

diffuse = module.analyze_localization(np.ones((32, 32), dtype=np.float32))
assert diffuse.bbox is None

print("Bonus A helper test: PASS")
