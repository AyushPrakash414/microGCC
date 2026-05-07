"""Quick demo script — trains 2 states and prints metrics."""
import sys, json
sys.path.insert(0, ".")

from app.pipelines.preprocessing import run_preprocessing_pipeline
from app.models.model_selector import train_and_evaluate_state

print("=" * 70)
print("STEP 1: Preprocessing data...")
print("=" * 70)
state_data = run_preprocessing_pipeline()
states_list = list(state_data.keys())
print(f"Total states available: {len(states_list)}")
print(f"States: {states_list[:10]}...")

demo_states = ["California", "Texas"]
all_results = {}

for state in demo_states:
    print("\n" + "=" * 70)
    print(f"STEP 2: Training ALL 4 models for {state}")
    print("=" * 70)
    result = train_and_evaluate_state(state, state_data[state])
    all_results[state] = result

    best = result["best_model"]
    print(f"\n>>> BEST MODEL for {state}: {best.upper()}")
    print(f"\n{'Model':<12} {'RMSE':>15} {'MAE':>15} {'MAPE (%)':>10} {'Time (s)':>10}")
    print("-" * 65)
    for model_name, info in result["models"].items():
        marker = " <-- BEST" if model_name == best else ""
        print(
            f"{model_name:<12} {info['rmse']:>15,.2f} {info['mae']:>15,.2f} "
            f"{info['mape']:>10.2f} {info['training_time_seconds']:>10.1f}{marker}"
        )

print("\n" + "=" * 70)
print("STEP 3: Testing Forecast Endpoint")
print("=" * 70)

from app.pipelines.forecasting_pipeline import run_forecasting_pipeline

for state in demo_states:
    forecast = run_forecasting_pipeline(state)
    print(f"\nForecast for {state} (model: {forecast['best_model']}):")
    print(f"{'Week':<6} {'Date':<12} {'Predicted Sales':>18} {'Lower Bound':>15} {'Upper Bound':>15}")
    print("-" * 70)
    for i, f in enumerate(forecast["forecast"], 1):
        print(
            f"  W{i:<3} {f['date']:<12} {f['predicted_sales']:>18,.2f} "
            f"{f['lower_bound']:>15,.2f} {f['upper_bound']:>15,.2f}"
        )

print("\n" + "=" * 70)
print("DONE! Models trained, evaluated, and forecasts generated.")
print("=" * 70)
