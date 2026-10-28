import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


def _valid_sample(sample):
    try:
        sol = np.asarray(sample["solution"])
        if sol.ndim != 2 or sol.size == 0 or sol.shape[0] > 30 or sol.shape[1] > 30:
            return False
        if not np.issubdtype(sol.dtype, np.integer):
            return False
        if sol.min() < 0 or sol.max() > 9:
            return False
        if not np.isfinite(sample["beam_score"]):
            return False
        if not len(sample["score_aug"]) or not np.all(np.isfinite(sample["score_aug"])):
            return False
        return True
    except Exception:
        return False


class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        try:
            self.valid_keys = set(dataset.keys)
        except Exception:
            self.valid_keys = None

    def load_decoded_results(self, store, run_name=""):
        if not os.path.isdir(store):
            print(f"*** No decoded results at {store}")
            return 0
        n_files = n_samples = n_bad = 0
        for key in os.listdir(store):
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                print(f"*** Skipping corrupt shard {key}: {e}")
                n_bad += 1
                continue
            n_files += 1
            base_key = key.split(".")[0]
            if self.valid_keys is not None and base_key not in self.valid_keys:
                n_bad += 1
                continue
            for i, sample in enumerate(outputs):
                if not _valid_sample(sample):
                    n_bad += 1
                    continue
                self.decoded_results.setdefault(base_key, {})[f"{key}{run_name}.out{i}"] = sample
                n_samples += 1
        print(f"*** Loaded {n_files} shards / {n_samples} samples from {store} (skipped {n_bad})")
        return n_samples

    def candidate_stats(self):
        """Per output key: number of unique candidate grids, number of samples, and the
        vote count / kgmon margin of the leading candidate (how sure the pool is)."""
        stats = {}
        for bk, v in self.decoded_results.items():
            groups = {}
            for g in v.values():
                groups.setdefault(hashable(g["solution"]), []).append(g)
            scored = sorted((getter_kgmon(gs) for gs in groups.values()), reverse=True)
            top_votes = max(len(gs) for gs in groups.values())
            margin = (scored[0] - scored[1]) if len(scored) > 1 else float("inf")
            stats[bk] = dict(unique=len(groups), samples=len(v), top_votes=top_votes, margin=margin)
        return stats

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")

