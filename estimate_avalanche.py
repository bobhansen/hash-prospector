from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple, TypeAlias
import enum
import sys

# Type alias for clarity: A 64-element 1D array or 64x64 2D array of floats
#    representing the probability that a flip in an input bit (x axis) will cause a
#    flip in an output bit (y axis) in an avalanche effect.
# For example, the operation [x XOR 0x01] would be represented as a 64x64 array where
#    the first column is all 1.0 (indicating that flipping the least significant bit
#    of the input always flips the least significant bit of the output), and all other
#    columns are 0.0.
AvalancheArray: TypeAlias = npt.NDArray[np.float64]
AvalancheArrayMeanP95: TypeAlias = Tuple[AvalancheArray, AvalancheArray, AvalancheArray]
Inputs: TypeAlias = npt.NDArray[np.uint64]  # 1D array of uint64 values

NBITS = 64

pool = ThreadPoolExecutor()

# Type alias for a hash operation: takes an integer and returns an integer
op: TypeAlias = Callable[[Inputs], Inputs]


def getRandomInputs(n: int) -> Inputs:
    """Generate n random uint64 inputs."""
    return np.random.randint(0, 2**64, size=n, dtype=np.uint64)


def calculate_avalanche(operation: op, n: int) -> AvalancheArrayMeanP95:
    """Calculate the actual avalanche property of an operation over n inputs.

    Returns a 64x64 array where element [i, j] represents the probability that
    flipping input bit i will cause output bit j to flip.
    """
    # Initialize avalanche matrix
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)

    # Parallelize over input bits using the pool
    def process_bit(input_bit: int) -> Tuple[int, npt.NDArray[np.float64]]:
        """Process a single input bit and return its avalanche row."""
        inputs = getRandomInputs(n)
        original_outputs = operation(inputs)

        # Create flipped inputs by XORing with a mask that flips this bit
        flipped_inputs = inputs ^ (np.uint64(1) << input_bit)
        flipped_outputs = operation(flipped_inputs)

        # Calculate which output bits flipped
        output_diffs = original_outputs ^ flipped_outputs

        # Count flips for each output bit
        row = np.zeros(NBITS, dtype=np.float64)
        for output_bit in range(NBITS):
            bit_mask = np.uint64(1) << output_bit
            flips = np.count_nonzero(output_diffs & bit_mask)
            row[output_bit] = flips / n

        return input_bit, row

    # Process all input bits in parallel
    results = pool.map(process_bit, range(NBITS))

    # Assemble results into the avalanche matrix
    for input_bit, row in results:
        avalanche[input_bit, :] = row

    # Calculate p95 confidence intervals using binomial proportion confidence interval
    # For each cell, we have n Bernoulli trials with observed probability p = avalanche[i,j]
    # Using normal approximation: std_error = sqrt(p * (1-p) / n)
    # 95% CI is approximately p ± 1.96 * std_error
    std_error = np.sqrt(avalanche * (1 - avalanche) / n)
    margin = 1.96 * std_error
    lower_p95 = np.clip(avalanche - margin, 0.0, 1.0)
    upper_p95 = np.clip(avalanche + margin, 0.0, 1.0)
    return [avalanche, lower_p95, upper_p95]


class BiasTarget(enum.Enum):
    RMS = enum.auto()
    MAX = enum.auto()


def identity(size: int) -> AvalancheArray:
    """Generate an identity matrix of given size."""
    return np.eye(size, dtype=np.float64)


def biases(arr: AvalancheArray) -> Tuple[float, float]:
    """Returns the (max bias, rms bias) of the given array."""
    return (
        float(np.max(np.abs(arr - 0.5))),
        float(np.sqrt(np.mean((arr - 0.5) ** 2)))
    )


class OpType(enum.Enum):
    """Enumeration of basic bit manipulation operations."""
    XOR = enum.auto()
    SHIFTR = enum.auto()
    SHIFTL = enum.auto()


def estimate_xor(_x: int) -> AvalancheArray:
    """Estimate the avalanche effect of an XOR operation with a constant.

    For XOR with a constant, flipping any input bit i will always flip output bit i
    (and only output bit i), regardless of the constant value.
    This is because (a ^ c) ^ ((a ^ (1 << i)) ^ c) = 1 << i for any constant c.

    Returns a 64x64 identity matrix where element [i, j] = 1.0 if i == j, else 0.0.
    """
    return identity(NBITS)


def test_xor(x: int, n: int) -> float:
    """Test the avalanche effect of an XOR operation with a constant."""
    def op_func(inputs: Inputs) -> Inputs:
        return inputs ^ np.uint64(x)

    estimated = estimate_xor(x)
    _, low_actual, high_actual = calculate_avalanche(op_func, n)

    # Calculate the percentage of estimated that were within the p95
    within_p95 = np.logical_and(
        estimated >= low_actual,
        estimated <= high_actual
    )

    return 100.0 * np.sum(within_p95) / (NBITS * NBITS)


def estimate_xor_shift(shift: int) -> AvalancheArray:
    """Estimate the avalanche effect of x XOR (x >> shift).
    
    Special case: When shift is 0, x ^ (x >> 0) = x ^ x = 0, so no bits ever flip.
    
    When an input bit i is flipped in the operation x ^ (x >> shift) for shift > 0:
    - Output bit i always flips (from the left side of XOR)
    - Output bit (i - shift) also flips if (i - shift) >= 0 (from the right side)
    
    This creates a 64x64 matrix where element [i, j] represents the probability
    that flipping input bit i will cause output bit j to flip.
    - Row i (input bit i), column i (output bit i) has 1.0
    - Row i also has 1.0 at column (i - shift) if that column exists
    """
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    
    # Special case: shift = 0 means x ^ x = 0, no bits flip
    if shift == 0:
        return avalanche
    
    for input_bit in range(NBITS):
        # Flipping input bit i always flips output bit i (from x in the XOR)
        avalanche[input_bit, input_bit] = 1.0
        
        # Flipping input bit i also flips output bit (i - shift) if valid
        # This comes from (x >> shift) in the XOR
        if input_bit >= shift:
            avalanche[input_bit, input_bit - shift] = 1.0
    
    return avalanche


def test_xor_shift(x: int, n: int) -> float:
    """Test the avalanche effect of an XOR operation with a constant."""
    def op_func(inputs: Inputs) -> Inputs:
        return inputs ^ (inputs >> np.uint64(x))

    estimated = estimate_xor_shift(x)
    _, low_actual, high_actual = calculate_avalanche(op_func, n)
    # print(f"Actual avalanche low p95 for XOR_SHIFT with {x:#018x}:\n{low_actual}")

    # Calculate the percentage of estimated that were within the p95
    within_p95 = np.logical_and(
        estimated >= low_actual,
        estimated <= high_actual
    )

    return 100.0 * np.sum(within_p95) / (NBITS * NBITS)



def main() -> None:
    n = 100000
    accuracies = list()
    for x in range(NBITS):
        accuracy = test_xor_shift(x, 100000)
        accuracies.append(accuracy)
        print(f"XOR_SHIFT with {x:#018x}: {accuracy:.2f}% of estimates within 95% confidence interval over {n} samples.")

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"xorshift with {x:#018x}: {average_accuracy:.2f}% of estimates within 95% confidence interval over {n} samples.")

    if average_accuracy < 100.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
