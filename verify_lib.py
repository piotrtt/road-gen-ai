import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    from src.main import generate_xodr_script
    print("[INFO] Successfully imported generate_xodr_script")

    # Call with dummy description
    script = generate_xodr_script("A long straight road")
    print(f"[INFO] Generated Script Element Count: {len(script)}")
    print("[INFO] Script Preview:")
    print(script[:200]) # First 200 chars

    # Verify components import
    from src.road_components.components.straight import generate_straight_road
    from src.road_components.definitions import StraightRoad
    
    comp = StraightRoad(id="test_road", length=100)
    code = generate_straight_road(comp)
    print(f"[INFO] Component Generator Output: {code}")

except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
except Exception as e:
    print(f"[ERROR] Execution failed: {e}")
