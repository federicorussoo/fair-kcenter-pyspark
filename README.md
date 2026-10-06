# Fair k-Center Clustering with PySpark

A PySpark MapReduce implementation of a coreset-based strategy for the **Fair k-Center Clustering** problem. This project is designed to handle large-scale geometric datasets across distributed clusters efficiently, enforcing strict demographic fairness constraints on the selected cluster centers.

## Core Algorithms

The solution extracts exactly `k_A` centers for demographic group `A` and `k_B` for group `B`, minimizing the maximum Euclidean distance from any point to its nearest center. It achieves this through two main components:

*   **Sequential Fair-FFT:** A variant of the Farthest-First Traversal (FFT) algorithm. It tracks demographic budgets dynamically, iteratively picking the furthest point from the current set of centers but restricting the selection to groups that have not yet exhausted their capacity.
*   **MR-Fair-FFT (MapReduce):** A 2-round distributed algorithm:
    *   **Round 1 (Map):** The dataset is partitioned into `L` chunks. The `Fair-FFT` algorithm runs independently on each partition in parallel to extract local coresets.
    *   **Round 2 (Reduce):** All local coresets are aggregated on the driver node. A final sequential `Fair-FFT` pass extracts the global set of `k` fair centers.

## Usage

### Prerequisites
* Python 3.x
* Apache Spark (PySpark)

### Execution
Run the application via the command line, providing the dataset path, the specific budgets for each demographic group (`k_A` and `k_B`), and the desired number of partitions (`L`) as the example below:

```bash
python main.py <path_to_dataset.csv> <k_A> <k_B> <L>
