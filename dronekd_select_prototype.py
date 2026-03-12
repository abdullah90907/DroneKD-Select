import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_DRONES = 12
NUM_ROUNDS = 50
TOP_K = 5

STATE_NAMES = {
    0: "normal",
    1: "moderate_link_drop",
    2: "severe_link_drop"
}


class Drone:
    def __init__(self, drone_id):
        self.drone_id = drone_id
        self.data_quality = np.clip(np.random.normal(0.72, 0.12), 0.35, 0.98)
        self.link_reliability = np.clip(np.random.normal(0.75, 0.15), 0.30, 0.99)
        self.energy = np.clip(np.random.normal(0.80, 0.10), 0.40, 1.00)
        self.freshness = 1.0
        self.base_contribution = np.clip(np.random.normal(0.68, 0.14), 0.25, 0.98)

    def evolve(self):
        self.link_reliability = np.clip(
            self.link_reliability + np.random.normal(0.0, 0.06), 0.20, 0.99
        )
        self.energy = np.clip(
            self.energy - np.random.uniform(0.005, 0.03), 0.15, 1.00
        )
        self.freshness = np.clip(
            self.freshness + np.random.uniform(0.02, 0.10), 0.20, 1.50
        )

    def simulate_local_distillation(self):
        communication_state = np.random.choice([0, 1, 2], p=[0.60, 0.28, 0.12])

        if communication_state == 0:
            link_factor = np.random.uniform(0.90, 1.00)
            comm_cost = np.random.uniform(0.8, 1.2)
            latency = np.random.uniform(0.8, 1.4)
        elif communication_state == 1:
            link_factor = np.random.uniform(0.65, 0.85)
            comm_cost = np.random.uniform(1.1, 1.7)
            latency = np.random.uniform(1.4, 2.3)
        else:
            link_factor = np.random.uniform(0.40, 0.62)
            comm_cost = np.random.uniform(1.5, 2.4)
            latency = np.random.uniform(2.3, 3.6)

        contribution_quality = np.clip(
            (
                0.45 * self.data_quality
                + 0.30 * self.base_contribution
                + 0.15 * self.energy
                + 0.10 * np.random.uniform(0.5, 1.0)
            )
            * link_factor,
            0.05,
            1.0,
        )

        success_prob = np.clip(
            0.50 * self.link_reliability + 0.25 * self.energy + 0.25 * link_factor,
            0.10,
            0.99,
        )
        success = np.random.rand() < success_prob

        return {
            "communication_state": communication_state,
            "contribution_quality": contribution_quality,
            "comm_cost": comm_cost,
            "latency": latency,
            "success": success,
        }


def contribution_aware_score(drone, local_result):
    score = (
        0.42 * local_result["contribution_quality"]
        + 0.33 * drone.link_reliability
        + 0.15 * drone.energy
        + 0.10 * min(drone.freshness, 1.0)
    )
    return score


def run_equal_weighting(drones, rounds=NUM_ROUNDS):
    global_accuracy = 0.42
    accuracy_curve = []
    communication_curve = []
    latency_curve = []
    energy_curve = []

    for _ in range(rounds):
        round_gain = 0.0
        round_comm = 0.0
        round_latency = 0.0
        selected = drones

        for drone in selected:
            local = drone.simulate_local_distillation()
            if local["success"]:
                round_gain += 0.010 * local["contribution_quality"]
                drone.freshness = 0.35
            round_comm += local["comm_cost"]
            round_latency += local["latency"]
            drone.evolve()

        global_accuracy = min(global_accuracy + round_gain, 0.94)

        accuracy_curve.append(global_accuracy)
        communication_curve.append(round_comm)
        latency_curve.append(round_latency / len(selected))
        energy_curve.append(np.mean([d.energy for d in drones]))

    return {
        "accuracy": accuracy_curve,
        "communication": communication_curve,
        "latency": latency_curve,
        "energy": energy_curve,
    }


def run_random_selection(drones, rounds=NUM_ROUNDS, top_k=TOP_K):
    global_accuracy = 0.42
    accuracy_curve = []
    communication_curve = []
    latency_curve = []
    energy_curve = []

    for _ in range(rounds):
        round_gain = 0.0
        round_comm = 0.0
        round_latency = 0.0
        selected = random.sample(drones, top_k)

        for drone in selected:
            local = drone.simulate_local_distillation()
            if local["success"]:
                round_gain += 0.014 * local["contribution_quality"]
                drone.freshness = 0.35
            round_comm += local["comm_cost"]
            round_latency += local["latency"]
            drone.evolve()

        for drone in drones:
            if drone not in selected:
                drone.evolve()

        global_accuracy = min(global_accuracy + round_gain, 0.95)

        accuracy_curve.append(global_accuracy)
        communication_curve.append(round_comm)
        latency_curve.append(round_latency / len(selected))
        energy_curve.append(np.mean([d.energy for d in drones]))

    return {
        "accuracy": accuracy_curve,
        "communication": communication_curve,
        "latency": latency_curve,
        "energy": energy_curve,
    }


def run_contribution_aware_selection(drones, rounds=NUM_ROUNDS, top_k=TOP_K):
    global_accuracy = 0.42
    accuracy_curve = []
    communication_curve = []
    latency_curve = []
    energy_curve = []
    selected_history = []

    for _ in range(rounds):
        scored = []

        for drone in drones:
            local = drone.simulate_local_distillation()
            score = contribution_aware_score(drone, local)
            scored.append((drone, local, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:top_k]
        selected_history.append([item[0].drone_id for item in selected])

        round_gain = 0.0
        round_comm = 0.0
        round_latency = 0.0

        weight_sum = sum(item[2] for item in selected) + 1e-8

        for drone, local, score in selected:
            normalized_weight = score / weight_sum
            if local["success"]:
                round_gain += 0.020 * local["contribution_quality"] * (1.0 + normalized_weight)
                drone.freshness = 0.30
            round_comm += local["comm_cost"]
            round_latency += local["latency"]
            drone.evolve()

        for drone, local, score in scored[top_k:]:
            drone.evolve()

        global_accuracy = min(global_accuracy + round_gain, 0.975)

        accuracy_curve.append(global_accuracy)
        communication_curve.append(round_comm)
        latency_curve.append(round_latency / max(len(selected), 1))
        energy_curve.append(np.mean([d.energy for d in drones]))

    return {
        "accuracy": accuracy_curve,
        "communication": communication_curve,
        "latency": latency_curve,
        "energy": energy_curve,
        "selected_history": selected_history,
    }


def clone_drones(base_drones):
    copied = []
    for d in base_drones:
        new_d = Drone(d.drone_id)
        new_d.data_quality = d.data_quality
        new_d.link_reliability = d.link_reliability
        new_d.energy = d.energy
        new_d.freshness = d.freshness
        new_d.base_contribution = d.base_contribution
        copied.append(new_d)
    return copied


def summarize_results(name, result):
    print(f"\n{name}")
    print(f"Final accuracy        : {result['accuracy'][-1]:.4f}")
    print(f"Average comm cost     : {np.mean(result['communication']):.4f}")
    print(f"Average latency       : {np.mean(result['latency']):.4f}")
    print(f"Final average energy  : {result['energy'][-1]:.4f}")


def save_accuracy_plot(equal_result, random_result, selective_result):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, equal_result["accuracy"], label="Equal weighting")
    plt.plot(rounds, random_result["accuracy"], label="Random selection")
    plt.plot(rounds, selective_result["accuracy"], label="Contribution aware selection")
    plt.xlabel("Training round")
    plt.ylabel("Global accuracy")
    plt.title("Global Accuracy Across Federated Distillation Rounds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_rounds.png"), dpi=160)
    plt.close()


def save_cost_plot(equal_result, random_result, selective_result):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, equal_result["communication"], label="Equal weighting")
    plt.plot(rounds, random_result["communication"], label="Random selection")
    plt.plot(rounds, selective_result["communication"], label="Contribution aware selection")
    plt.xlabel("Training round")
    plt.ylabel("Communication cost")
    plt.title("Communication Cost Across Rounds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "communication_cost_vs_rounds.png"), dpi=160)
    plt.close()


def save_latency_plot(equal_result, random_result, selective_result):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, equal_result["latency"], label="Equal weighting")
    plt.plot(rounds, random_result["latency"], label="Random selection")
    plt.plot(rounds, selective_result["latency"], label="Contribution aware selection")
    plt.xlabel("Training round")
    plt.ylabel("Average round latency")
    plt.title("Latency Across Rounds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "latency_vs_rounds.png"), dpi=160)
    plt.close()


def save_energy_plot(equal_result, random_result, selective_result):
    rounds = np.arange(1, NUM_ROUNDS + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, equal_result["energy"], label="Equal weighting")
    plt.plot(rounds, random_result["energy"], label="Random selection")
    plt.plot(rounds, selective_result["energy"], label="Contribution aware selection")
    plt.xlabel("Training round")
    plt.ylabel("Average residual energy")
    plt.title("Average Drone Energy Across Rounds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "energy_vs_rounds.png"), dpi=160)
    plt.close()


def save_bar_chart(equal_result, random_result, selective_result):
    methods = [
        "Equal weighting",
        "Random selection",
        "Contribution aware"
    ]
    final_accuracy = [
        equal_result["accuracy"][-1],
        random_result["accuracy"][-1],
        selective_result["accuracy"][-1]
    ]
    avg_comm = [
        np.mean(equal_result["communication"]),
        np.mean(random_result["communication"]),
        np.mean(selective_result["communication"])
    ]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, final_accuracy, width, label="Final accuracy")
    plt.bar(x + width / 2, avg_comm, width, label="Average comm cost")
    plt.xticks(x, methods)
    plt.title("Final Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_comparison.png"), dpi=160)
    plt.close()


def main():
    print("\nStarting DroneKD Select prototype...\n")

    base_drones = [Drone(i) for i in range(NUM_DRONES)]

    equal_drones = clone_drones(base_drones)
    random_drones = clone_drones(base_drones)
    selective_drones = clone_drones(base_drones)

    equal_result = run_equal_weighting(equal_drones)
    random_result = run_random_selection(random_drones)
    selective_result = run_contribution_aware_selection(selective_drones)

    summarize_results("Equal weighting baseline", equal_result)
    summarize_results("Random selection baseline", random_result)
    summarize_results("Contribution aware selective distillation", selective_result)

    acc_gain_vs_equal = (
        (selective_result["accuracy"][-1] - equal_result["accuracy"][-1]) /
        max(equal_result["accuracy"][-1], 1e-8)
    ) * 100.0

    acc_gain_vs_random = (
        (selective_result["accuracy"][-1] - random_result["accuracy"][-1]) /
        max(random_result["accuracy"][-1], 1e-8)
    ) * 100.0

    print(f"\nAccuracy gain over equal weighting : {acc_gain_vs_equal:.2f}%")
    print(f"Accuracy gain over random selection: {acc_gain_vs_random:.2f}%")

    save_accuracy_plot(equal_result, random_result, selective_result)
    save_cost_plot(equal_result, random_result, selective_result)
    save_latency_plot(equal_result, random_result, selective_result)
    save_energy_plot(equal_result, random_result, selective_result)
    save_bar_chart(equal_result, random_result, selective_result)

    print("\nSaved result figures in the results folder:")
    print("1. accuracy_vs_rounds.png")
    print("2. communication_cost_vs_rounds.png")
    print("3. latency_vs_rounds.png")
    print("4. energy_vs_rounds.png")
    print("5. final_comparison.png")
    print("\nPrototype completed successfully.")


if __name__ == "__main__":
    main()