import json
import sys


def validate_json_file(file_path):
    try:
        with open(file_path, "r") as f:
            states = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
        return False

    if not isinstance(states, list):
        print("Error: JSON file should contain a list of states")
        return False

    # Track all PCs seen across all states
    all_pcs_seen = set()
    backpressure_count = 0
    last_decoded_pcs = []

    for state_idx, state in enumerate(states):
        # Check ActiveList size
        active_list = state.get("ActiveList", [])
        if len(active_list) > 32:
            print(
                f"Error: ActiveList size {len(active_list)} exceeds maximum of 32 in state {state_idx}"
            )
            return False

        # Check FreeList size
        free_list = state.get("FreeList", [])
        if len(free_list) > 32:
            print(
                f"Error: FreeList size {len(free_list)} exceeds maximum of 32 in state {state_idx}"
            )
            return False

        # Check IntegerQueue size
        integer_queue = state.get("IntegerQueue", [])
        if len(integer_queue) > 32:
            print(
                f"Error: IntegerQueue size {len(integer_queue)} exceeds maximum of 32 in state {state_idx}"
            )
            return False

        # Check DecodedPCs
        decoded_pcs = state.get("DecodedPCs", [])
        if decoded_pcs is not None and not (
            len(decoded_pcs) == 4 or len(decoded_pcs) == 0
        ):
            print(
                f"Error: DecodedPCs size {len(decoded_pcs)} is not 4 or 0 in state {state_idx}"
            )
            return False

        # Collect PCs from this state's ActiveList
        active_list_pcs = [entry.get("PC", -1) for entry in active_list]
        all_pcs_seen.update(active_list_pcs)
        if last_decoded_pcs == decoded_pcs and len(decoded_pcs) > 0:
            backpressure_count += 1
            print(f"Backpressure detected in state {state_idx}")
        last_decoded_pcs = decoded_pcs

    # Check PC coverage (0 to 39) across all states
    missing_pcs = [pc for pc in range(40) if pc not in all_pcs_seen]
    if missing_pcs:
        print(
            f"Error: The following PCs are missing from all ActiveLists: {missing_pcs}"
        )
        return False
    if backpressure_count == 0:
        print("No backpressure detected")
        return False

    print("All validation checks passed successfully!")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate.py <json_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not validate_json_file(file_path):
        sys.exit(1)
