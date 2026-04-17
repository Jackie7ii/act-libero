import os
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from libero.libero import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_goal",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "task_embeddings"),
    )
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    model = SentenceTransformer(args.model_name)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()

    task_emb_dict = {}
    for i in range(task_suite.n_tasks):
        desc = task_suite.get_task(i).language
        task_emb_dict[desc] = model.encode(desc)
        print(f"{i}: {desc}")

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.task_suite}_task_embeddings.npy")
    np.save(save_path, task_emb_dict)
    print(f"saved to {save_path}")


if __name__ == "__main__":
    main()
