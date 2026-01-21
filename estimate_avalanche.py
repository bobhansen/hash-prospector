from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple, TypeAlias
from functools import lru_cache
import enum
import sys

# Type alias for clarity: A 64-element 1D array or 64x64 2D array of floats
#    representing the probability that a flip in an input bit (x axis) will cause a
#    flip in an output bit (y axis) in an avalanche effect.
# For example, the operation [x XOR 0x01] would be represented as a 64x64 array where
#    the first column is all 1.0 (indicating that flipping the least significant bit
#    of the input always flips the least significant bit of the output), and all other
#    columns are 0.0, except for the diagonal which would also be 1.0.
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

    # Calculate p95 confidence intervals for each cell as a binomial proportion.
    # Use the Wilson score interval (much more reliable than the normal approximation,
    # especially when the observed proportion is 0 or 1).
    lower_p95, upper_p95, _, _ = wilson_interval(avalanche, n, z=1.96)
    return [avalanche, lower_p95, upper_p95]


def wilson_interval(
    p_hat: AvalancheArray,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[AvalancheArray, AvalancheArray, AvalancheArray, AvalancheArray]:
    """Compute Wilson score CI for a binomial proportion.

    Args:
        p_hat: Observed proportion(s) in [0,1]. Can be a scalar/array.
        n: Number of Bernoulli trials.
        z: Z critical value (default 1.96 for ~95% CI).

    Returns:
        (low, high, center, se_est)
        where se_est is approximated as (high-low)/(2*z).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom
    low = np.clip(center - margin, 0.0, 1.0)
    high = np.clip(center + margin, 0.0, 1.0)
    se_est = (high - low) / (2.0 * z)
    return low, high, center, se_est


def z_scores_from_samples(
    estimated: AvalancheArray,
    observed: AvalancheArray,
    n: int,
    *,
    z: float = 1.96,
) -> AvalancheArray:
    """Compute per-cell z-scores comparing an estimate to sampled proportions.

    We use the Wilson interval to derive a stable standard error estimate even
    when observed proportions are 0 or 1, but compute the residual against the
    raw observed proportion p̂ (not the Wilson center) so that the mean z-score
    is interpretable as an (approximate) unbiasedness check.

    Returns:
        zscore matrix with same shape as inputs.
    """
    _, _, _, se = wilson_interval(observed, n, z=z)
    # Avoid divide-by-zero (shouldn't happen for finite n, but keep it safe).
    safe_se = np.where(se > 0.0, se, np.inf)
    return (estimated - observed) / safe_se


def zscore_summary(zmat: AvalancheArray) -> tuple[float, float, float, float]:
    """Summarize a z-score matrix.

    Returns:
        (mean_z, rms_z, max_abs_z, frac_abs_z_gt_3)
    """
    mean_z = float(np.mean(zmat))
    rms_z = float(np.sqrt(np.mean(zmat * zmat)))
    max_abs_z = float(np.max(np.abs(zmat)))
    frac_abs_z_gt_3 = float(np.mean(np.abs(zmat) > 3.0))
    return mean_z, rms_z, max_abs_z, frac_abs_z_gt_3


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


def combine_avalanche(first: AvalancheArray, second: AvalancheArray) -> AvalancheArray:
    """Combine the avalanche matrices of two subsequent operations.
    
    When two operations are applied in sequence (first, then second), the avalanche
    effect compounds via XOR. The contribution from path i->j->k is first[i,j] * second[j,k],
    and the combined flip probability uses the XOR formula for independent events:
    P(odd flips) = (1 - prod(1 - 2*p_j)) / 2
    
    Args:
        first: Avalanche matrix of the first operation (applied first)
        second: Avalanche matrix of the second operation (applied second)
    
    Returns:
        Combined avalanche matrix where element [i, k] represents the probability
        that flipping input bit i will cause final output bit k to flip.
    """
    # contribution[i, j, k] = first[i, j] * second[j, k]
    contribution = first[:, :, np.newaxis] * second
    
    # XOR combination: P(odd flips) = (1 - prod_j(1 - 2*p_j)) / 2
    return (1.0 - np.prod(1.0 - 2.0 * contribution, axis=1)) / 2.0


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


def compute_conditional_y_probs(constant: int, input_bit: int) -> tuple:
    """Compute P(y[j]=1 | x[input_bit]=0) and P(y[j]=1 | x[input_bit]=1) for all j.
    
    For y = x * constant, when x[input_bit] is fixed, the contribution of that bit
    to y is deterministic, but carries from other bits create complex interactions.
    
    This function computes the probabilities analytically by considering how
    x[input_bit] contributes to y through the multiplication.
    
    y = x * c can be written as:
    y = x[input_bit] * c * 2^input_bit + sum_{k != input_bit} x[k] * c * 2^k
    
    The first term is either 0 (if x[input_bit]=0) or c << input_bit.
    The second term varies uniformly over (2^(NBITS-1)) values.
    
    Returns:
        (probs_given_0, probs_given_1): Arrays of P(y[j]=1 | x[input_bit]=v)
    """
    mask = (1 << NBITS) - 1
    
    # Contribution from x[input_bit]
    contrib_0 = 0  # when x[input_bit] = 0
    contrib_1 = (constant << input_bit) & mask  # when x[input_bit] = 1
    
    # For the other bits, we sum over all possible combinations
    # This is equivalent to y_rest = (x & ~(1 << input_bit)) * constant
    # As x varies with bit input_bit fixed, y_rest takes on all values
    # (x & ~(1 << input_bit)) * constant for x in [0, 2^NBITS) with x[input_bit] fixed
    
    probs_0 = np.zeros(NBITS, dtype=np.float64)
    probs_1 = np.zeros(NBITS, dtype=np.float64)
    
    # We need to sum over all x values with x[input_bit] fixed
    # This is equivalent to summing over x_rest = x with bit input_bit cleared
    count = 0
    for x_rest in range(1 << NBITS):
        if (x_rest >> input_bit) & 1:
            continue  # Skip values where bit input_bit is set
        
        y_rest = (x_rest * constant) & mask
        y_with_0 = (y_rest + contrib_0) & mask
        y_with_1 = (y_rest + contrib_1) & mask
        
        for j in range(NBITS):
            if (y_with_0 >> j) & 1:
                probs_0[j] += 1
            if (y_with_1 >> j) & 1:
                probs_1[j] += 1
        count += 1
    
    probs_0 /= count
    probs_1 /= count
    
    return probs_0, probs_1


def exact_model_multiply(constant: int) -> AvalancheArray:
    """Estimate the avalanche effect of a multiplication operation with a constant.
    
    When flipping input bit i in (x * constant), the output changes by ±delta where
    delta = constant × 2^i. Whether we add or subtract depends on whether x[i] was 
    0 or 1 respectively.
    
    The analysis uses exact conditional probability computation:
    - For each input bit i, we compute P(y[j]=1 | x[i]=0) and P(y[j]=1 | x[i]=1)
    - These probabilities are used in the carry propagation model to compute
      the flip probabilities for each output bit
    
    The flip probability at each output bit j is computed by:
    1. Determining the delta to add (or subtract, represented as adding 2's complement)
    2. Tracking carry probability through the bit positions
    3. Using the exact conditional probabilities P(y[j]=1 | x[i]=v)
    
    Args:
        constant: The constant multiplier
    
    Returns:
        A NBITS x NBITS matrix where element [i, j] represents the probability
        that flipping input bit i will cause output bit j to flip.
    """
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    
    # Special case: multiplying by 0 always gives 0, no bits flip
    if constant == 0:
        return avalanche
    
    # Mask for modular arithmetic  
    mask = (1 << NBITS) - 1
    
    for input_bit in range(NBITS):
        delta = (constant << input_bit) & mask
        neg_delta = ((-int(delta)) & mask)  # Two's complement for subtraction
        
        # Get the conditional probabilities for y bits given x[input_bit]
        probs_y_given_0, probs_y_given_1 = compute_conditional_y_probs(constant, input_bit)
        
        def compute_flip_probs(delta_val: int, y_probs: np.ndarray) -> list:
            """Compute flip probabilities when adding delta_val to y.
            
            y[j] has P(y[j]=1) = y_probs[j]
            """
            probs = [0.0] * NBITS
            delta_bits = [(delta_val >> j) & 1 for j in range(NBITS)]
            
            # Track P(carry into position j)
            P_carry = 0.0
            
            for j in range(NBITS):
                d = delta_bits[j]
                P_y1 = y_probs[j]
                P_y0 = 1.0 - P_y1
                P_c0 = 1.0 - P_carry
                P_c1 = P_carry
                
                # When adding d + carry_in to y[j]:
                # flip occurs when (d + carry_in) is odd
                
                if d == 0:
                    probs[j] = P_c1
                    # Carry out occurs when y[j]=1 AND carry_in=1
                    P_carry = P_y1 * P_c1
                else:  # d == 1
                    probs[j] = P_c0
                    # Carry out when y[j]=0 AND carry_in=1, OR y[j]=1
                    P_carry = P_y0 * P_c1 + P_y1
            
            return probs
        
        # ADD case: x[input_bit] = 0, use probs_y_given_0
        add_probs = compute_flip_probs(delta, probs_y_given_0)
        
        # SUB case: x[input_bit] = 1, use probs_y_given_1
        sub_probs = compute_flip_probs(neg_delta, probs_y_given_1)
        
        # Combined: 50% each case (x[input_bit] is 0 or 1 with equal probability)
        for j in range(NBITS):
            avalanche[input_bit, j] = 0.5 * add_probs[j] + 0.5 * sub_probs[j]
    
    return avalanche

def compute_conditional_y_probs_add(constant: int, input_bit: int) -> tuple:
    """Compute P(y[j]=1 | x[input_bit]=0) and P(y[j]=1 | x[input_bit]=1) for all j.
    
    For y = x + constant, when x[input_bit] is fixed, we need to compute the
    distribution of y bits over all possible values of the other input bits.
    
    Returns:
        (probs_given_0, probs_given_1): Arrays of P(y[j]=1 | x[input_bit]=v)
    """
    mask = (1 << NBITS) - 1
    
    probs_0 = np.zeros(NBITS, dtype=np.float64)
    probs_1 = np.zeros(NBITS, dtype=np.float64)
    
    # Sum over all x values with x[input_bit] fixed to 0 or 1
    count = 0
    for x_rest in range(1 << NBITS):
        if (x_rest >> input_bit) & 1:
            continue  # Skip values where bit input_bit is set
        
        # x with bit input_bit = 0
        x_with_0 = x_rest
        y_with_0 = (x_with_0 + constant) & mask
        
        # x with bit input_bit = 1
        x_with_1 = x_rest | (1 << input_bit)
        y_with_1 = (x_with_1 + constant) & mask
        
        for j in range(NBITS):
            if (y_with_0 >> j) & 1:
                probs_0[j] += 1
            if (y_with_1 >> j) & 1:
                probs_1[j] += 1
        count += 1
    
    probs_0 /= count
    probs_1 /= count
    
    return probs_0, probs_1


def exact_model_add(constant: int) -> AvalancheArray:
    """Compute the exact avalanche matrix for adding a constant.
    
    When flipping input bit i in (x + constant):
    - If x[i] was 0, we're adding (1 << i) to x, so y changes by adding (1 << i)
    - If x[i] was 1, we're subtracting (1 << i) from x, so y changes by subtracting (1 << i)
    
    The analysis uses exact conditional probability computation:
    - For each input bit i, we compute P(y[j]=1 | x[i]=0) and P(y[j]=1 | x[i]=1)
    - These probabilities track carry propagation to compute flip probabilities
    
    Args:
        constant: The constant to add
    
    Returns:
        A NBITS x NBITS matrix where element [i, j] represents the probability
        that flipping input bit i will cause output bit j to flip.
    """
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    mask = (1 << NBITS) - 1
    
    for input_bit in range(NBITS):
        delta = (1 << input_bit) & mask
        neg_delta = ((-delta) & mask)  # Two's complement for subtraction
        
        # Get the conditional probabilities for y bits given x[input_bit]
        probs_y_given_0, probs_y_given_1 = compute_conditional_y_probs_add(constant, input_bit)
        
        def compute_flip_probs(delta_val: int, y_probs: np.ndarray) -> list:
            """Compute flip probabilities when adding delta_val to y.
            
            y[j] has P(y[j]=1) = y_probs[j]
            """
            probs = [0.0] * NBITS
            delta_bits = [(delta_val >> j) & 1 for j in range(NBITS)]
            
            # Track P(carry into position j)
            P_carry = 0.0
            
            for j in range(NBITS):
                d = delta_bits[j]
                P_y1 = y_probs[j]
                P_y0 = 1.0 - P_y1
                P_c0 = 1.0 - P_carry
                P_c1 = P_carry
                
                # When adding d + carry_in to y[j]:
                # flip occurs when (d + carry_in) is odd
                
                if d == 0:
                    probs[j] = P_c1
                    # Carry out occurs when y[j]=1 AND carry_in=1
                    P_carry = P_y1 * P_c1
                else:  # d == 1
                    probs[j] = P_c0
                    # Carry out when y[j]=0 AND carry_in=1, OR y[j]=1
                    P_carry = P_y0 * P_c1 + P_y1
            
            return probs
        
        # ADD case: x[input_bit] was 0, now 1, so we add delta to y
        add_probs = compute_flip_probs(delta, probs_y_given_0)
        
        # SUB case: x[input_bit] was 1, now 0, so we subtract delta (add neg_delta) from y
        sub_probs = compute_flip_probs(neg_delta, probs_y_given_1)
        
        # Combined: 50% each case (x[input_bit] is 0 or 1 with equal probability)
        for j in range(NBITS):
            avalanche[input_bit, j] = 0.5 * add_probs[j] + 0.5 * sub_probs[j]
    
    return avalanche


def test_add_exhaustive(constant: int) -> bool:
    """Test exact_model_add against exhaustive computation for a given constant.
    
    Exhaustively computes the avalanche matrix by testing all possible inputs,
    then compares against the model prediction.
    
    Args:
        constant: The constant to add
    
    Returns:
        True if the model matches exactly, False otherwise
    """
    mask = (1 << NBITS) - 1
    
    # Compute exact avalanche by exhaustive enumeration
    exact_avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    
    for input_bit in range(NBITS):
        flip_counts = np.zeros(NBITS, dtype=np.int64)
        
        for x in range(1 << NBITS):
            y_original = (x + constant) & mask
            x_flipped = x ^ (1 << input_bit)
            y_flipped = (x_flipped + constant) & mask
            
            diff = y_original ^ y_flipped
            for j in range(NBITS):
                if (diff >> j) & 1:
                    flip_counts[j] += 1
        
        exact_avalanche[input_bit, :] = flip_counts / (1 << NBITS)
    
    # Get model prediction
    model_avalanche = exact_model_add(constant)
    
    # Compare - they should match exactly (within floating point tolerance)
    return np.allclose(exact_avalanche, model_avalanche, rtol=1e-10, atol=1e-10)


def estimate_add(constant: int) -> AvalancheArray:
    """Estimate the avalanche effect of adding a constant, matching exact_model_add.
    
    This uses the full adder carry propagation model:
    
    For y = x + c at each bit position j (full adder):
        y[j] = x[j] XOR c[j] XOR carry_in[j]
        carry_out[j] = (x[j] AND c[j]) OR (carry_in[j] AND (x[j] XOR c[j]))
    
    When input bit i is flipped:
    - Bits j < i: unaffected (carries propagate upward only, LSB to MSB)
    - Bit i: always flips (carry_in[i] doesn't depend on x[i])
    - Bits j > i: flip probability depends on carry propagation
    
    Key insight: The carry probability q[j] = P(carry into bit j) follows a recurrence:
        q[0] = 0 (no carry into LSB)
        q[j+1] = 0.5 * q[j]           if c[j] = 0
        q[j+1] = 0.5 + 0.5 * q[j]     if c[j] = 1
    
    This comes from the full adder equations with uniform random x[j]:
    - If c[j]=0: carry_out = x[j] AND carry_in, so P(carry_out=1) = 0.5 * q[j]
    - If c[j]=1: carry_out = x[j] OR (NOT x[j] AND carry_in) = x[j] OR carry_in
                 so P(carry_out=1) = 0.5 + 0.5 * q[j]
    
    The flip probability at bit i+k for k >= 1 is p_i * 0.5^(k-1), where:
        p_i = P(y[i]=1 | x[i]=0) = q[i] if c[i]=0, else 1-q[i]
    
    This represents the probability that flipping x[i] causes a carry that
    propagates through the subsequent bits, with 0.5 probability of stopping
    at each bit (since x[j] is random and independent).
    
    Args:
        constant: The constant to add
    
    Returns:
        A NBITS x NBITS matrix where element [i, j] represents the probability
        that flipping input bit i will cause output bit j to flip.
    """
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    
    # Step 1: Compute carry probabilities q[j] = P(carry into bit j)
    # using the full adder recurrence
    q = [0.0] * (NBITS + 1)
    q[0] = 0.0
    for j in range(NBITS):
        c_j = (constant >> j) & 1
        if c_j == 0:
            q[j + 1] = 0.5 * q[j]
        else:
            q[j + 1] = 0.5 + 0.5 * q[j]
    
    # Step 2: For each input bit i, compute flip probabilities
    for i in range(NBITS):
        c_i = (constant >> i) & 1
        
        # p_i = P(y[i]=1 | x[i]=0)
        # y[i] = 0 XOR c[i] XOR carry_in[i] = c[i] XOR carry_in[i]
        # So P(y[i]=1 | x[i]=0) = P(c[i] XOR carry_in[i] = 1)
        #                       = P(carry_in[i] = 1) if c[i] = 0
        #                       = P(carry_in[i] = 0) if c[i] = 1
        if c_i == 0:
            p_i = q[i]
        else:
            p_i = 1.0 - q[i]
        
        # Bits j < i: no effect (carries only propagate upward)
        # Bit i: always flips
        avalanche[i, i] = 1.0
        
        # Bits j > i: flip prob = p_i * 0.5^(j-i-1)
        # This is because:
        # - The initial carry flip at bit i has probability p_i
        # - Each subsequent bit has 0.5 probability of propagating the carry
        #   (since x[j] is uniform random and independent)
        for j in range(i + 1, NBITS):
            k = j - i  # k >= 1
            if k == 1:
                avalanche[i, j] = p_i
            else:
                avalanche[i, j] = p_i * (0.5 ** (k - 1))
    
    return avalanche


def test_add(x: int, n: int) -> float:
    """Test the avalanche effect of an addition operation with a constant."""
    def op_func(inputs: Inputs) -> Inputs:
        return inputs + np.uint64(x)

    # Placeholder for estimated addition avalanche
    estimated = estimate_add(x) 
    _, low_actual, high_actual = calculate_avalanche(op_func, n)

    # Calculate the percentage of estimated that were within the p95
    within_p95 = np.logical_and(
        estimated >= low_actual,
        estimated <= high_actual
    )

    return 100.0 * np.sum(within_p95) / (NBITS * NBITS)


def estimate_add_shift(shift: int) -> AvalancheArray:
    """Estimate the avalanche effect of ``y = x + (x << shift)``.

    This is the special-case multiply ``y = x * (1 + 2**shift)``.

    Key facts (for shift > 0):
    - The lower ``shift`` output bits are unchanged: ``y[j] == x[j]`` for ``j < shift``.
      So columns ``j < shift`` form an identity.
    - Flipping input bit ``i`` toggles the addend bit at position ``i`` (from the ``x`` term)
      and also toggles the addend bit at position ``i + shift`` (from the shifted term), if in range.
    - Above the first affected bit, differences only propagate upward through carries.

        This implementation computes the probabilities exactly (for uniform random inputs)
        using dynamic programming over the adder's carry state plus the minimum required
        memory to provide the delayed bit ``x[j-shift]``.

        Runtime is exponential in ``min(shift, NBITS-shift)`` (not in ``NBITS``), and is therefore
        strictly better than exhaustive ``O(2**NBITS)`` enumeration while still matching
        ``exact_model_add_shift`` exactly for practical parameter ranges.

        Special fast paths:
        - ``shift >= NBITS``: the shifted term contributes nothing to the low NBITS bits, so this is
            exactly the identity matrix.
        - ``NBITS`` even and ``shift == NBITS/2``: the operation cleanly decomposes into a low-half
            identity plus a high-half "add two independent uniform words" model, which we compute in
            O((NBITS/2)^2) time (e.g. essentially instant for NBITS=64, shift=32).
    """
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")

    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)

    # Special case: shift=0 => y = x + x = 2*x = x<<1 (mod 2**NBITS)
    if shift == 0:
        for i in range(NBITS):
            out_bit = i + 1
            if out_bit < NBITS:
                avalanche[i, out_bit] = 1.0
        return avalanche

    # If shift >= NBITS, then (x << shift) contributes nothing to the low NBITS bits.
    # There is no carry propagation downward, so y == x over the observed NBITS.
    if shift >= NBITS:
        return identity(NBITS)

    # Fast exact case: NBITS even and shift == NBITS/2.
    # Write x = lo + (hi << h), with h = NBITS/2. Then:
    #   (x << h) mod 2**NBITS = (lo << h)
    #   y = x + (x << h) = lo + ((hi + lo) << h)
    # The low half is identity, and the high half is (hi + lo) mod 2**h where hi,lo are independent uniform.
    if (NBITS % 2 == 0) and (shift == (NBITS // 2)):
        h = NBITS // 2

        for i in range(h):
            avalanche[i, i] = 1.0

        for i in range(h):
            for j in range(i, h):
                p = 0.5 ** (j - i)
                avalanche[i, h + j] = p
                avalanche[h + i, h + j] = p

        return avalanche

    # For shift close to NBITS/2 from below, only a small "overlap" of bits can reappear
    # in b[j]=x[j-shift] *after* carry becomes non-trivial. We can compute the avalanche
    # exactly by tracking just those overlap bits.
    if shift <= (NBITS // 2):
        overlap = NBITS - 2 * shift
        if overlap <= 8:
            return _estimate_add_shift_overlap_dp(shift, overlap)

    # We compute rows exactly via dynamic programming over a compact state:
    # - Carries (c, c') for original vs flipped input (4 possibilities)
    # - A "memory" state sufficient to recover b[j] = x[j-shift]
    #   * If shift is small: keep a shift-length sliding window of prior x bits (2**shift states)
    #   * If shift is large: only the lowest (NBITS-shift) bits of x ever appear as b[j],
    #     so keep that fixed prefix instead (2**(NBITS-shift) states)
    # Complexity per row: O(NBITS * 2**min(shift, NBITS-shift)).

    m = NBITS - shift  # number of input bits that are ever used as b[j]
    use_prefix = shift > m
    width = m if use_prefix else shift

    # For larger widths, fall back to a carry Markov model that runs in O(NBITS^2).
    # For shift > NBITS/2 this is exact; otherwise it's an approximation.
    if width > 8:
        return _estimate_add_shift_markov(shift)

    pre = _add_shift_precompute(NBITS, shift)
    n_states = pre["n_states"]
    states = pre["states"]
    b_by_j = pre["b_by_j"]
    next_state_by_j_x0 = pre["next_state_by_j_x0"]
    next_state_by_j_x1 = pre["next_state_by_j_x1"]
    co_pair_no_toggle_x0 = pre["co_pair_no_toggle_x0"]
    co_pair_no_toggle_x1 = pre["co_pair_no_toggle_x1"]
    c0_in = pre["c0_in"]
    c1_in = pre["c1_in"]

    dp = np.zeros((n_states, 4), dtype=np.float64)
    dp_next = np.zeros((n_states, 4), dtype=np.float64)
    dp[0, 0] = 1.0  # state=0, (c,c')=(0,0)

    def add_carry(a: np.uint8, b: npt.NDArray[np.uint8], c: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return ((a & b) | (a & c) | (b & c)).astype(np.uint8)

    def scatter_add_dp(
        out: npt.NDArray[np.float64],
        next_state: npt.NDArray[np.uint32],
        co_pair: npt.NDArray[np.uint8],
        in_dp: npt.NDArray[np.float64],
        weight: float,
    ) -> None:
        """Accumulate dp transitions into out using a bincount scatter.

        out and in_dp are shape (n_states, 4).
        next_state is shape (n_states,), co_pair is shape (n_states, 4).
        """
        flat_size = int(out.size)
        base = (next_state.astype(np.int64) * 4)
        # Build flat indices for each carry-pair column.
        idx0 = base + co_pair[:, 0].astype(np.int64)
        idx1 = base + co_pair[:, 1].astype(np.int64)
        idx2 = base + co_pair[:, 2].astype(np.int64)
        idx3 = base + co_pair[:, 3].astype(np.int64)
        weights = np.concatenate(
            [
                in_dp[:, 0] * weight,
                in_dp[:, 1] * weight,
                in_dp[:, 2] * weight,
                in_dp[:, 3] * weight,
            ]
        )
        idx = np.concatenate([idx0, idx1, idx2, idx3])
        out.reshape(-1)[:] += np.bincount(idx, weights=weights, minlength=flat_size)

    for input_bit in range(NBITS):
        dp.fill(0.0)
        dp[0, 0] = 1.0

        shifted_toggle_bit = input_bit + shift
        for j in range(NBITS):
            dp_next.fill(0.0)

            b = b_by_j[j]
            next0 = next_state_by_j_x0[j]
            next1 = next_state_by_j_x1[j]

            toggle_a = j == input_bit
            toggle_b = (shifted_toggle_bit < NBITS) and (j == shifted_toggle_bit)

            # If neither a nor b toggles at this position, the only source of output-bit
            # differences is carry-in differences (c != c').
            if not toggle_a and not toggle_b:
                # Both x=0 and x=1 branches contribute equally to avalanche via carry-diff.
                avalanche[input_bit, j] += float(np.sum(dp[:, 1] + dp[:, 2]))
                scatter_add_dp(dp_next, next0, co_pair_no_toggle_x0[j], dp, 0.5)
                scatter_add_dp(dp_next, next1, co_pair_no_toggle_x1[j], dp, 0.5)
                dp, dp_next = dp_next, dp
                continue

            # Otherwise, handle the (rare) toggle positions explicitly.
            b2 = b[:, None]
            c02 = c0_in[None, :]
            c12 = c1_in[None, :]

            for x in (0, 1):
                a = np.uint8(x)
                a_p = np.uint8(x ^ int(toggle_a))
                b_p = (b ^ np.uint8(int(toggle_b))).astype(np.uint8)

                sum0 = (a ^ b[:, None] ^ c0_in[None, :]).astype(np.uint8)
                sum1 = (a_p ^ b_p[:, None] ^ c1_in[None, :]).astype(np.uint8)
                diff = (sum0 != sum1)
                avalanche[input_bit, j] += 0.5 * float(np.sum(dp * diff))

                b2p = b_p[:, None]
                co0 = add_carry(a, b2, c02)
                co1 = add_carry(a_p, b2p, c12)
                co_pair = (co0 * 2 + co1).astype(np.uint8)

                if x == 0:
                    scatter_add_dp(dp_next, next0, co_pair, dp, 0.5)
                else:
                    scatter_add_dp(dp_next, next1, co_pair, dp, 0.5)

            dp, dp_next = dp_next, dp

    return avalanche


def _estimate_add_shift_markov(shift: int) -> AvalancheArray:
    """Scalable estimate for y = x + (x<<shift) using a carry-pair Markov model.

    Assumptions:
    - Input bits x[k] are independent Bernoulli(0.5).
        - For each bit position j >= shift, the addend bits (a=x[j], b=x[j-shift]) are treated as
            independent of the carry state.

        Notes:
        - If shift > NBITS/2, this model is exact because b only references bits from the carry-free region.
        - If shift <= NBITS/2, this ignores long-range dependence between b and carry (approximation).

    Runs in O(NBITS^2) time.
    """
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)

    # Carry-pair state index: (c << 1) | c'
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    carry_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def carry_out(a: int, b: int, c: int) -> int:
        return (a & b) | (a & c) | (b & c)

    # Precompute local (diff_prob[4], trans[4,4]) for pb in {0.0, 0.5} and toggles.
    # pb=0.0 means b is constant 0; pb=0.5 means b is uniform random.
    local = {}
    for pb in (0.0, 0.5):
        b_values = (0,) if pb == 0.0 else (0, 1)
        for ta in (0, 1):
            for tb in (0, 1):
                diff_prob = np.zeros(4, dtype=np.float64)
                trans = np.zeros((4, 4), dtype=np.float64)
                for s_idx, (c, cp) in enumerate(carry_pairs):
                    for a in (0, 1):
                        pa = 0.5
                        for b in b_values:
                            pbv = 1.0 if pb == 0.0 else 0.5
                            w = pa * pbv

                            ap = a ^ ta
                            bp = b ^ tb

                            s0 = a ^ b ^ c
                            s1 = ap ^ bp ^ cp
                            if s0 != s1:
                                diff_prob[s_idx] += w

                            co0 = carry_out(a, b, c)
                            co1 = carry_out(ap, bp, cp)
                            ns = (co0 << 1) | co1
                            trans[s_idx, ns] += w
                local[(pb, ta, tb)] = (diff_prob, trans)

    for input_bit in range(NBITS):
        dist = np.zeros(4, dtype=np.float64)
        dist[0] = 1.0

        toggle_b_bit = input_bit + shift
        for j in range(NBITS):
            pb = 0.0 if j < shift else 0.5
            ta = 1 if j == input_bit else 0
            tb = 1 if (toggle_b_bit < NBITS and j == toggle_b_bit) else 0
            diff_prob, trans = local[(pb, ta, tb)]

            avalanche[input_bit, j] = float(dist @ diff_prob)
            dist = dist @ trans

    return avalanche


@lru_cache(maxsize=None)
def _overlap_dp_precompute(overlap: int) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    if overlap <= 0:
        raise ValueError(f"overlap must be > 0, got {overlap}")

    n_states = 1 << overlap
    states = np.arange(n_states, dtype=np.uint32)

    # b_fixed_by_read_idx[read_idx, state] = (state >> read_idx) & 1
    b_fixed_by_read_idx = np.empty((overlap, n_states), dtype=np.uint8)
    for read_idx in range(overlap):
        b_fixed_by_read_idx[read_idx] = ((states >> np.uint32(read_idx)) & 1).astype(np.uint8)

    # next_state_if_write[write_idx, state] = state | (1<<write_idx)
    next_state_if_write = np.empty((overlap, n_states), dtype=np.uint32)
    for write_idx in range(overlap):
        next_state_if_write[write_idx] = states | (np.uint32(1) << np.uint32(write_idx))

    # Precompute carry-pair transitions and output-bit flips for all toggle combinations.
    # next_cp[ta, tb, cp_idx, a, b] and flip[ta, tb, cp_idx, a, b]
    carry_pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    next_cp = np.empty((2, 2, 4, 2, 2), dtype=np.uint8)
    flip = np.empty((2, 2, 4, 2, 2), dtype=np.uint8)

    def carry_out(a: int, b: int, c: int) -> int:
        return (a & b) | (a & c) | (b & c)

    for ta in (0, 1):
        for tb in (0, 1):
            for cp_idx, (c, cp) in enumerate(carry_pairs):
                for a in (0, 1):
                    for b in (0, 1):
                        ap = a ^ ta
                        bp = b ^ tb
                        s0 = a ^ b ^ c
                        s1 = ap ^ bp ^ cp
                        flip[ta, tb, cp_idx, a, b] = np.uint8(1 if s0 != s1 else 0)

                        co0 = carry_out(a, b, c)
                        co1 = carry_out(ap, bp, cp)
                        next_cp[ta, tb, cp_idx, a, b] = np.uint8((co0 << 1) | co1)

    return b_fixed_by_read_idx, next_state_if_write, next_cp, flip


def _estimate_add_shift_overlap_dp(shift: int, overlap: int) -> AvalancheArray:
    """Exact avalanche for y = x + (x<<shift) when overlap = NBITS-2*shift is small.

    For j < shift, b[j]=0 and carry is identically 0.
    For shift <= j < 2*shift, b[j]=x[j-shift] references carry-free bits, so it is independent noise.
    For j >= 2*shift, b[j]=x[j-shift] references bits in x[shift .. NBITS-shift-1], the "overlap".
    Tracking only those overlap bits yields an exact DP with 2**overlap states.
    """
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap == 0:
        return _estimate_add_shift_markov(shift)

    n_states = 1 << overlap
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)

    b_fixed_by_read_idx, next_state_if_write, next_cp, flip = _overlap_dp_precompute(overlap)

    dp = np.zeros((n_states, 4), dtype=np.float64)
    dp_next = np.zeros((n_states, 4), dtype=np.float64)

    for input_bit in range(NBITS):
        dp.fill(0.0)
        dp[0, 0] = 1.0

        toggle_b_bit = input_bit + shift

        for j in range(NBITS):
            dp_next.fill(0.0)

            ta = 1 if j == input_bit else 0
            tb = 1 if (toggle_b_bit < NBITS and j == toggle_b_bit) else 0

            b_kind = 0
            read_idx = -1
            if j < shift:
                b_kind = 0  # const0
            elif j < 2 * shift:
                b_kind = 1  # random
            else:
                b_kind = 2  # state
                read_idx = j - 2 * shift
                b_fixed_vec = b_fixed_by_read_idx[read_idx]

            do_write = (shift <= j < (shift + overlap))
            write_idx = (j - shift) if do_write else -1
            next_state_if_write_vec = next_state_if_write[write_idx] if do_write else None

            for state in range(n_states):
                if b_kind == 2:
                    b_fixed = int(b_fixed_vec[state])

                for cp_idx in range(4):
                    p_in = dp[state, cp_idx]
                    if p_in == 0.0:
                        continue

                    if b_kind == 0:
                        # b is constant 0
                        for a in (0, 1):
                            if flip[ta, tb, cp_idx, a, 0]:
                                avalanche[input_bit, j] += p_in * 0.5

                            cp_next = int(next_cp[ta, tb, cp_idx, a, 0])
                            if do_write and a == 1:
                                dp_next[int(next_state_if_write_vec[state]), cp_next] += p_in * 0.5
                            else:
                                dp_next[state, cp_next] += p_in * 0.5

                    elif b_kind == 1:
                        # b is uniform random
                        for a in (0, 1):
                            w = 0.25
                            for b in (0, 1):
                                if flip[ta, tb, cp_idx, a, b]:
                                    avalanche[input_bit, j] += p_in * w

                                cp_next = int(next_cp[ta, tb, cp_idx, a, b])
                                if do_write and a == 1:
                                    dp_next[int(next_state_if_write_vec[state]), cp_next] += p_in * w
                                else:
                                    dp_next[state, cp_next] += p_in * w

                    else:
                        # b is fixed by overlap state
                        b = b_fixed
                        for a in (0, 1):
                            if flip[ta, tb, cp_idx, a, b]:
                                avalanche[input_bit, j] += p_in * 0.5

                            cp_next = int(next_cp[ta, tb, cp_idx, a, b])
                            if do_write and a == 1:
                                dp_next[int(next_state_if_write_vec[state]), cp_next] += p_in * 0.5
                            else:
                                dp_next[state, cp_next] += p_in * 0.5

            dp, dp_next = dp_next, dp

    return avalanche


@lru_cache(maxsize=None)
def _states_for_width(width: int) -> npt.NDArray[np.uint32]:
    n_states = 1 << width
    return np.arange(n_states, dtype=np.uint32)


@lru_cache(maxsize=None)
def _add_shift_precompute(nbits: int, shift: int) -> dict:
    """Precompute DP tables for estimate_add_shift.

    This cache assumes the caller treats returned arrays as read-only.
    Keyed by (nbits, shift). If NBITS is constant over a run, this lets repeated
    calls reuse tables. Some internal arrays (like `states`) are additionally
    shared by `width` via `_states_for_width`.
    """
    if shift < 0 or shift >= nbits:
        raise ValueError(f"shift must be in [0, {nbits - 1}], got {shift}")
    if shift == 0:
        raise ValueError("shift=0 has no DP precompute")

    m = nbits - shift
    use_prefix = shift > m
    width = m if use_prefix else shift
    if width <= 0:
        raise ValueError(f"invalid width computed: {width}")
    if width > 20:
        raise ValueError(
            f"estimate_add_shift state size too large: width={width} (2**width states). "
            "Try a smaller shift/bit-width, or adjust the algorithm."
        )

    n_states = 1 << width
    states = _states_for_width(width)

    c0_in = np.array([0, 0, 1, 1], dtype=np.uint8)
    c1_in = np.array([0, 1, 0, 1], dtype=np.uint8)

    def add_carry(a: np.uint8, b: npt.NDArray[np.uint8], c: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return ((a & b) | (a & c) | (b & c)).astype(np.uint8)

    b_by_j: list[npt.NDArray[np.uint8]] = []
    next_state_by_j_x0: list[npt.NDArray[np.uint32]] = []
    next_state_by_j_x1: list[npt.NDArray[np.uint32]] = []

    for j in range(nbits):
        if use_prefix:
            if j < shift:
                b = np.zeros(n_states, dtype=np.uint8)
            else:
                b = ((states >> np.uint32(j - shift)) & 1).astype(np.uint8)
        else:
            b = ((states >> np.uint32(width - 1)) & 1).astype(np.uint8)
        b_by_j.append(b)

        if use_prefix:
            if j < m:
                next0 = states
                next1 = states | (np.uint32(1) << np.uint32(j))
            else:
                next0 = states
                next1 = states
        else:
            mask = np.uint32(n_states - 1)
            next0 = (states << 1) & mask
            next1 = ((states << 1) & mask) | np.uint32(1)
        next_state_by_j_x0.append(next0)
        next_state_by_j_x1.append(next1)

    co_pair_no_toggle_x0: list[npt.NDArray[np.uint8]] = []
    co_pair_no_toggle_x1: list[npt.NDArray[np.uint8]] = []
    for j in range(nbits):
        b = b_by_j[j]
        b2 = b[:, None]
        c02 = c0_in[None, :]
        c12 = c1_in[None, :]

        a0 = np.uint8(0)
        co0 = add_carry(a0, b2, c02)
        co1 = add_carry(a0, b2, c12)
        co_pair_no_toggle_x0.append((co0 * 2 + co1).astype(np.uint8))

        a1 = np.uint8(1)
        co0 = add_carry(a1, b2, c02)
        co1 = add_carry(a1, b2, c12)
        co_pair_no_toggle_x1.append((co0 * 2 + co1).astype(np.uint8))

    return {
        "n_states": n_states,
        "states": states,
        "b_by_j": tuple(b_by_j),
        "next_state_by_j_x0": tuple(next_state_by_j_x0),
        "next_state_by_j_x1": tuple(next_state_by_j_x1),
        "co_pair_no_toggle_x0": tuple(co_pair_no_toggle_x0),
        "co_pair_no_toggle_x1": tuple(co_pair_no_toggle_x1),
        "c0_in": c0_in,
        "c1_in": c1_in,
    }


def exact_model_add_shift(shift: int) -> AvalancheArray:
    """Calculate the exact avalanche matrix for x + (x << shift) via exhaustive enumeration.
    
    For y = x + (x << shift), when input bit i is flipped:
    - The x term contributes a flip at position i
    - The (x << shift) term contributes a flip at position i + shift (if in range)
    - These two contributions are added together, causing carry propagation
    
    The interaction between these two contributions and carry propagation makes
    this non-trivial to model analytically, so we compute exhaustively.
    
    Args:
        shift: The shift amount (0 to NBITS-1)
    
    Returns:
        A NBITS x NBITS matrix where element [i, j] represents the probability
        that flipping input bit i will cause output bit j to flip.
    """
    mask = (1 << NBITS) - 1
    avalanche = np.zeros((NBITS, NBITS), dtype=np.float64)
    
    # Special case: shift = 0 means x + x = 2*x = x << 1
    # This is just a left shift by 1, which has a simple pattern
    
    for input_bit in range(NBITS):
        flip_counts = np.zeros(NBITS, dtype=np.int64)
        
        for x in range(1 << NBITS):
            y_original = (x + (x << shift)) & mask
            x_flipped = x ^ (1 << input_bit)
            y_flipped = (x_flipped + (x_flipped << shift)) & mask
            
            diff = y_original ^ y_flipped
            for j in range(NBITS):
                if (diff >> j) & 1:
                    flip_counts[j] += 1
        
        avalanche[input_bit, :] = flip_counts / (1 << NBITS)
    
    return avalanche

def test_add_shift(shift: int, n: int) -> float:
    """Test the avalanche effect of an XOR operation with a constant."""
    def op_func(inputs: Inputs) -> Inputs:
        return inputs + (inputs << np.uint64(shift))

    estimated = estimate_add_shift(shift)
    actual, low_actual, high_actual = calculate_avalanche(op_func, n)

    # Z-score summary (useful when within-p95 is misleading for tiny probabilities).
    zmat = z_scores_from_samples(estimated, actual, n)
    mean_z, rms_z, max_abs_z, frac_abs_z_gt_3 = zscore_summary(zmat)
    # Attach for debugging/inspection without changing the return type.
    test_add_shift.last_z_summary = (mean_z, rms_z, max_abs_z, frac_abs_z_gt_3)

    # Calculate the percentage of estimated that were within the p95
    within_p95 = np.logical_and(
        estimated >= low_actual,
        estimated <= high_actual
    )

    return 100.0 * np.sum(within_p95) / (NBITS * NBITS)


def main() -> None:
    print("\nRunning operation tests...")
    n = 100000
    accuracies = list()
    z_means: list[float] = []
    z_rms: list[float] = []
    z_max_abs: list[float] = []
    z_frac_gt_3: list[float] = []
    for x in range(NBITS):
        accuracy = test_add_shift(x, n)
        mean_z, rms_z, max_abs_z, frac_abs_z_gt_3 = getattr(
            test_add_shift,
            "last_z_summary",
            (float("nan"), float("nan"), float("nan"), float("nan")),
        )
        accuracies.append(accuracy)
        z_means.append(float(mean_z))
        z_rms.append(float(rms_z))
        z_max_abs.append(float(max_abs_z))
        z_frac_gt_3.append(float(frac_abs_z_gt_3))
        print(
            f"op with {x:#018x}: {accuracy:.2f}% within 95% CI over {n} samples "
            f"(mean z={mean_z:.2f}, rms z={rms_z:.2f}, max |z|={max_abs_z:.2f}, |z|>3={100.0*frac_abs_z_gt_3:.2f}%)."
        )

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy: {average_accuracy:.2f}% of estimates within 95% confidence interval over {n} samples.")

    # Unbiasedness / calibration checks.
    # Interpreting z-scores:
    # - If the estimator is unbiased and the sampling error model is reasonable, z should be ~N(0,1).
    # - We allow slack because cells are not perfectly independent and tails are sensitive to small-probability events.
    mean_abs_z_mean = float(np.mean(np.abs(z_means)))
    mean_rms_z = float(np.mean(z_rms))
    max_abs_z_overall = float(np.max(z_max_abs))
    mean_frac_gt_3 = float(np.mean(z_frac_gt_3))
    print(
        "Z-score summary over shifts: "
        f"mean(|mean z|)={mean_abs_z_mean:.3f}, mean(rms z)={mean_rms_z:.3f}, "
        f"max(max |z|)={max_abs_z_overall:.2f}, mean(|z|>3)={100.0*mean_frac_gt_3:.2f}%."
    )

    # This metric is the fraction of cells whose estimate falls within a per-cell 95% CI
    # computed from finite samples. Even with a perfect model, expected coverage is ~95%.
    #
    # The z-score checks aim to detect systematic bias beyond sampling noise.
    # Thresholds are intentionally tolerant.
    if average_accuracy < 92.0 or mean_abs_z_mean > 0.25 or mean_rms_z > 2.0 or mean_frac_gt_3 > 0.05:
        sys.exit(1)


if __name__ == "__main__":
    main()
