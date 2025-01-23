from typing import Optional
import matplotlib.pyplot as plt


def plot_accuracy_bar_chart(
    context_accuracy_dict: dict,
    title: str = "Accuracy by Number of Contexts",
    xlabel: str = "Number of Contexts",
    ylabel: str = "Match Rate (%)",
    bar_color: str = "skyblue",
    show_values: bool = True,
    save_path: Optional[str] = None,
) -> None:
    # Convert keys to strings (categorical labels)
    contexts = [str(key) for key in context_accuracy_dict.keys()]
    accuracies = [
        round(value * 100.0, 2) for value in context_accuracy_dict.values()
    ]  # Convert to percentages and round

    plt.figure(figsize=(10, 6))
    bars = plt.bar(contexts, accuracies, color=bar_color, edgecolor="black")

    # Adding titles and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(range(0, 101, 10), fontsize=12)
    plt.ylim(0, max(accuracies) + 10)  # Add some space above the highest bar

    # Adding value labels on top of each bar
    if show_values:
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{accuracy}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)  # High resolution (300 DPI)
        print(f"Plot saved as {save_path}")

    plt.show()
