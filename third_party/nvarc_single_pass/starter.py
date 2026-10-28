import os
import sys
import time
import json
import torch
import argparse
import traceback
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, test_path, marker_dir):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Serialize the unsloth import/patching across workers (baseline behaviour),
    # but never wait forever for a dead predecessor.
    if rank > 0:
        waited = 0
        while not os.path.exists(os.path.join(marker_dir, f"worker{rank-1}")) and waited < 900:
            time.sleep(5)
            waited += 5

    from arc_solver import worker

    with open(os.path.join(marker_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    attempts = 0
    while attempts < 2 and time.time() < end_time:
        attempts += 1
        try:
            worker(rank, queue, end_time, test_path=test_path)
            break
        except Exception as e:
            print(f"[Rank {rank}] worker crashed ({type(e).__name__}: {e}); attempt {attempts}")
            traceback.print_exc()
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            if attempts >= 2:
                print(f"[Rank {rank}] giving up.")

    print(f"[Rank {rank}] done!")


def estimated_work(task):
    """Rough cost proxy: tokens of all train pairs (TTT + context) plus an estimate of
    the decoded output sizes. Cheap tasks first => the deadline truncates the most
    expensive (and least likely to be solved) tail."""
    def ntok(g):
        return len(g) * (len(g[0]) + 1)
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return train_tokens * 16 + test_tokens * 8 * len(task["test"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--nprocs", type=int, default=0)
    parser.add_argument("--order", type=str, default="cheap", choices=["cheap", "sorted", "file"])
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--marker-dir", type=str, default="/kaggle/working/markers")
    args, _ = parser.parse_known_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if args.test_path:
        test_path = args.test_path
    elif rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    if not os.path.exists(test_path):
        test_path = test_path.replace("/kaggle/input/competitions/", "/kaggle/input/")

    with open(test_path, "r") as f:
        data = json.load(f)

    if args.keys_file:
        with open(args.keys_file) as f:
            keys = [k for k in json.load(f) if k in data]
    else:
        keys = sorted(data.keys())
        if not rerun_mode and os.getenv("ARC_ALL_KEYS", "") != "1":
            debug_keys = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
            keys = [k for k in keys if k in debug_keys]

    if args.order == "cheap":
        keys = sorted(keys, key=lambda k: estimated_work(data[k]))
    elif args.order == "sorted":
        keys = sorted(keys)

    nprocs = args.nprocs or min(4, torch.cuda.device_count())
    os.makedirs(args.marker_dir, exist_ok=True)
    for f_ in os.listdir(args.marker_dir):
        try:
            os.remove(os.path.join(args.marker_dir, f_))
        except Exception:
            pass

    print(f"[starter] {len(keys)} tasks, {nprocs} workers, order={args.order}, "
          f"budget={(args.end_time - time.time())/60:.1f} min, test_path={test_path}")

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    try:
        mp.spawn(local_worker, args=(queue, args.end_time, test_path, args.marker_dir), nprocs=nprocs)
    except Exception as e:
        print(f"[starter] spawn finished with error: {type(e).__name__}: {e}")
        traceback.print_exc()
    print("[starter] finished.")

