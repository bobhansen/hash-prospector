import numpy as np
import numpy.typing as npt
from typing import List, Tuple, TypeAlias
import enum
import subprocess

# Type alias for clarity: A 64-element 1D array or 64x64 2D array of floats
FloatArray: TypeAlias = npt.NDArray[np.float64]


class Target(enum.Enum):
    RMS = enum.auto()
    MAX = enum.auto()


def biases(arr: FloatArray) -> Tuple[float, float]:
    """Returns the (max bias, rms bias) of the given array."""
    return (
        float(np.max(np.abs(arr - 0.5))),
        float(np.sqrt(np.mean((arr - 0.5) ** 2)))
    )

def prob_xor(p1: float, p2: float) -> float:
    """
    Computes the probabilistic XOR sum: p1 + p2 - 2*p1*p2.
    """
    return p1 + p2 - 2.0 * p1 * p2

def prob_xor_reduce(arr: FloatArray, axis: int = 0) -> FloatArray:
    """
    Reduces an array along a given axis using the prob_xor operator.
    """
    # Initialize result with the first element along the axis
    # The slicing ensures we maintain correct dimensions during iteration
    result: FloatArray = arr[0] if axis == 0 else arr[:, 0]
    
    dim: int = arr.shape[axis]
    
    # Iterate through the remaining elements along the specified axis
    for i in range(1, dim):
        current_slice: FloatArray = arr[i] if axis == 0 else arr[:, i]
        # We manually apply the scalar logic element-wise via numpy operations
        # p1 + p2 - 2*p1*p2 works directly on numpy arrays
        result = result + current_slice - (2.0 * result * current_slice)
        
    return result

def calculate_carry_vector(row_idx: int, active_shifts: List[int], n: int = 64) -> FloatArray:
    """
    Calculates the probability that the carry entering 'row_idx' flips 
    given a flip in input bit 'i'.
    """
    carry_probs: FloatArray = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        contributions: List[float] = []
        
        for s in active_shifts:
            p: int = i + s
            
            # The disturbance must occur strictly below the current row 
            if p < row_idx:
                distance: int = row_idx - p
                # Geometric decay: Prob = 0.5^distance
                contributions.append(0.5 ** distance)
        
        if contributions:
            p_final: float = contributions[0]
            for p_next in contributions[1:]:
                p_final = prob_xor(p_final, p_next)
            carry_probs[i] = p_final
            
    return carry_probs

def optimize_multiplication_constant(input_avalanche_matrix: FloatArray, target: Target) -> Tuple[int, FloatArray]:
    """
    Generates a 64-bit constant C that optimizes the strict avalanche criterion (SAC).
    
    Args:
        input_avalanche_matrix: 64x64 matrix of probabilities (0.0 to 1.0).
        
    Returns:
        A tuple containing:
        - The optimized 64-bit constant (int)
        - The resulting 64x64 output avalanche matrix (FloatArray)
    """
    n: int = 64
    final_c_int: int = 0
    active_shifts: List[int] = [] 
    
    output_matrix: FloatArray = np.zeros((n, n), dtype=np.float64)
    
    for r in range(n):
        # 1. Calculate Base Carry Vector
        base_carry: FloatArray = calculate_carry_vector(r, active_shifts, n)
        
        # 2. Formulate Candidates
        # Candidate 0: C_r = 0
        row_cand_0: FloatArray = base_carry.copy()
        
        # Candidate 1: C_r = 1
        row_cand_1: FloatArray = base_carry.copy()
        # Apply deterministic flip at index 0 (Input bit 0 aligned with Output bit r)
        # Note: We must explicitly cast to float for type safety, though numpy handles it
        row_cand_1[0] = prob_xor(float(row_cand_1[0]), 1.0)
        
        # 3. Multiply Candidates by Input Matrix A using XOR-Sum
        # We perform element-wise multiplication with broadcasting
        # row_cand[:, None] is shape (64, 1), input_matrix is (64, 64)
        # Resulting weighted matrices are (64, 64)
        weighted_0: FloatArray = row_cand_0[:, np.newaxis] * input_avalanche_matrix
        weighted_1: FloatArray = row_cand_1[:, np.newaxis] * input_avalanche_matrix
        
        # Reduce along the rows (axis 0) to get the final row vector (shape 64,)
        final_row_0: FloatArray = prob_xor_reduce(weighted_0, axis=0)
        final_row_1: FloatArray = prob_xor_reduce(weighted_1, axis=0)
        
        # 4. Evaluate Cost (max, RMS Bias from 0.5)
        biases_0 = biases(final_row_0)
        biases_1 = biases(final_row_1)

        if r == 0:
            use_1 = True  # Force LSB to 1 for invertibility
        elif target == Target.MAX:
            if biases_0[0] == biases_1[0]:
                use_1 = biases_0[1] > biases_1[1]
            else:
                use_1 = biases_0[0] > biases_1[0]
        elif target == Target.RMS:
            use_1 = biases_0[1] > biases_1[1]
        else:
            raise ValueError("Unknown target type for optimization.")   
 
        # 5. Select Best Bit
        if use_1:
            final_c_int |= (1 << r)
            active_shifts.append(r)
            output_matrix[r] = final_row_1
        else:
            output_matrix[r] = final_row_0
            
    return final_c_int, output_matrix


def optimize_multiplication_constant2(input_avalanche_matrix: FloatArray, target: Target) -> Tuple[int, FloatArray]:
    """
    Generates a 64-bit constant C that optimizes the SAC.
    
    Improved Logic:
    Uses a 'Lookahead' cost. It evaluates not just the current output row,
    but also the quality (entropy) of the Carry Vector that will be passed 
    to the next row. This prevents the '0x5555' oscillation by valuing 
    future avalanche potential.
    """
    n: int = 64
    final_c_int: int = 0
    active_shifts: List[int] = [] 
    
    output_matrix: FloatArray = np.zeros((n, n), dtype=np.float64)
    
    print(f"Optimizing Constant (N={n})...")
    
    for r in range(n):
        # --- step A: Evaluate Current Row 'r' ---
        
        # 1. Calculate Base Carry entering row 'r' (from previous decisions)
        base_carry_in = calculate_carry_vector(r, active_shifts, n)
        
        # 2. Candidate 0 (C_r = 0): Just the carry
        row_cand_0 = base_carry_in.copy()
        
        # 3. Candidate 1 (C_r = 1): Carry + Direct Injection at index 0
        row_cand_1 = base_carry_in.copy()
        row_cand_1[0] = prob_xor(float(row_cand_1[0]), 1.0)
        
        # 4. Multiply by Matrix A to get actual Output Probabilities
        w_0 = row_cand_0[:, np.newaxis] * input_avalanche_matrix
        w_1 = row_cand_1[:, np.newaxis] * input_avalanche_matrix
        
        out_row_0 = prob_xor_reduce(w_0, axis=0)
        out_row_1 = prob_xor_reduce(w_1, axis=0)
        
        # Cost 1: Current Row Bias (L2 Norm)
        cost_curr_0 = np.sum((out_row_0 - 0.5)**2)
        cost_curr_1 = np.sum((out_row_1 - 0.5)**2)
        
        # --- Step B: Evaluate Future Carry (Lookahead to r+1) ---
        
        # If we choose 0, active shifts don't change.
        # We simulate the carry vector for row r+1 based on CURRENT active_shifts
        next_carry_0 = calculate_carry_vector(r + 1, active_shifts, n)
        
        # If we choose 1, we ADD 'r' to active shifts temporarily
        next_carry_1 = calculate_carry_vector(r + 1, active_shifts + [r], n)
        
        # Evaluate quality of these future carries (Projected through Matrix A)
        # We want the carry itself to be close to 0.5 to maximize future mixing.
        # Note: We multiply by A to see the *effective* carry entropy.
        
        # (Optimization: We can just evaluate the raw carry vector entropy 
        # because A is unitary-ish, but multiplying is safer).
        wc_0 = next_carry_0[:, np.newaxis] * input_avalanche_matrix
        wc_1 = next_carry_1[:, np.newaxis] * input_avalanche_matrix
        
        future_vec_0 = prob_xor_reduce(wc_0, axis=0)
        future_vec_1 = prob_xor_reduce(wc_1, axis=0)
        
        cost_future_0 = np.sum((future_vec_0 - 0.5)**2)
        cost_future_1 = np.sum((future_vec_1 - 0.5)**2)
        
        # --- Step C: Combined Decision ---
        
        total_cost_0 = cost_curr_0 + cost_future_0
        total_cost_1 = cost_curr_1 + cost_future_1
        
        if total_cost_1 < total_cost_0:
            final_c_int |= (1 << r)
            active_shifts.append(r)
            output_matrix[r] = out_row_1
        else:
            output_matrix[r] = out_row_0
            
    return final_c_int, output_matrix

def apply_multiplication_constant(constant: int, input_avalanche_matrix: FloatArray) -> FloatArray:
    """
    Calculates the output avalanche matrix resulting from multiplying the input
    by a specific 64-bit integer constant 'constant'.
    
    Args:
        constant: The 64-bit integer multiplier.
        input_avalanche_matrix: 64x64 matrix of probabilities.
        
    Returns:
        The resulting 64x64 output avalanche matrix.
    """
    n: int = 64
    active_shifts: List[int] = [] 
    output_matrix: FloatArray = np.zeros((n, n), dtype=np.float64)
    
    # Iterate row by row (LSB to MSB)
    for r in range(n):
        # 1. Determine the bit value of the constant at this position
        bit_val: int = (constant >> r) & 1
        
        # 2. Calculate Base Carry Vector (Noise from lower bits of C)
        # This vector represents the flip prob of bit 'r' due ONLY to carries 
        # from the previously processed bits (active_shifts).
        row_vector: FloatArray = calculate_carry_vector(r, active_shifts, n)
        
        # 3. If the constant bit is 1, we add the direct injection term
        if bit_val == 1:
            # We add a deterministic flip (Prob 1.0) at index 0.
            # This represents the current input bit shifting directly into the current output position.
            row_vector[0] = prob_xor(float(row_vector[0]), 1.0)
            
            # Update state for future rows: this bit will now contribute to carries
            active_shifts.append(r)
            
        # 4. Multiply by Input Matrix A using XOR-Sum
        # We perform element-wise multiplication with broadcasting to weight the columns
        # Shape: (64, 64)
        weighted_matrix: FloatArray = row_vector[:, np.newaxis] * input_avalanche_matrix
        
        # Reduce along the rows (axis 0) to get the final row vector (shape 64,)
        final_row: FloatArray = prob_xor_reduce(weighted_matrix, axis=0)
        
        # 5. Store result
        output_matrix[r] = final_row
            
    return output_matrix

def apply_xor_shift(shift: int, input_avalanche_matrix: FloatArray) -> FloatArray:
    """
    Calculates the output avalanche matrix resulting from the operation:
        Y = X ^ (X >> shift)
    
    Args:
        shift: The right-shift amount (integer).
        input_avalanche_matrix: 64x64 matrix of probabilities.
        
    Returns:
        The resulting 64x64 output avalanche matrix.
    """
    n: int = 64
    
    # If shift is 0, Y = X ^ X = 0. This destroys information.
    # However, strictly speaking, X ^ (X >> 0) is 0. 
    # Usually in hashing contexts, if shift is 0, it's a no-op, but 
    # strictly following the formula it zeros the state.
    # If the user meant "rotate", that's different, but standard xorshift 
    # with shift 0 is usually invalid or handled as identity.
    # Here we assume strict formula adherence:
    if shift == 0:
        return np.zeros((n, n), dtype=np.float64)
        
    if shift >= n:
        # If shifting beyond width, X >> shift is 0.
        # So Y = X ^ 0 = X. 
        return input_avalanche_matrix.copy()

    # Initialize output with the probabilities of X (the left operand of XOR)
    output_matrix: FloatArray = input_avalanche_matrix.copy()
    
    # 1. Get the shifted rows (the right operand of XOR: X >> shift)
    #    The probability row 'i' of (X >> shift) corresponds to 
    #    row 'i + shift' of the original input.
    shifted_part: FloatArray = input_avalanche_matrix[shift:, :]
    
    # 2. Determine the affected rows in the output
    #    Only indices 0 to (63 - shift) have a matching bit in (X >> shift).
    target_rows_slice = slice(0, n - shift)
    
    # 3. Apply Probabilistic XOR Sum
    #    We combine the original row 'i' with the shifted row 'i+shift'.
    original_part: FloatArray = output_matrix[target_rows_slice]
    
    # Vectorized calculation: p1 + p2 - 2*p1*p2
    combined_part: FloatArray = (
        original_part + shifted_part - (2.0 * original_part * shifted_part)
    )
    
    # 4. Update the matrix
    output_matrix[target_rows_slice] = combined_part
    
    return output_matrix

def optimize_xor_shift(input_avalanche_matrix: FloatArray, target: Target) -> Tuple[int, FloatArray]:
    """
    Finds the optimal right-shift constant 'c' for the operation Y = X ^ (X >> c)
    to minimize the maximum bias of the output avalanche matrix.
    
    Args:
        input_avalanche_matrix: 64x64 matrix of probabilities.
        
    Returns:
        A tuple containing:
        - The optimal shift amount 'c' (int)
        - The resulting 64x64 output avalanche matrix (FloatArray)
    """
    n: int = 64
    best_c: int = -1
    best_biases: Tuple[float, float] = (float('inf'), float('inf'))
    best_matrix: FloatArray = np.zeros((n, n), dtype=np.float64)
    
    # Iterate through valid shifts (1 to 63)
    for c in range(1, n):
        candidate_matrix = apply_xor_shift(c, input_avalanche_matrix)
        current_biases = biases(candidate_matrix)        

        # Update best
        if target == Target.MAX:
            if current_biases[0] == best_biases[0]:
                new_best = current_biases[1] < best_biases[1]
            else:
                new_best = current_biases[0] < best_biases[0]
        elif target == Target.RMS:
            new_best = current_biases[1] < best_biases[1]

        if new_best:
            best_biases = current_biases
            best_c = c
            # We copy to ensure we store the actual matrix state, not a reference
            best_matrix = candidate_matrix.copy()
            
    return best_c, best_matrix

def find_high_bias_coordinates(avalanche_matrix: FloatArray, target_bias: float) -> List[Tuple[int, int]]:
    """
    Finds all coordinates in the avalanche matrix where the bias exceeds the target.
    
    Args:
        avalanche_matrix: 64x64 matrix of probabilities.
        target_bias: The threshold bias value (absolute deviation from 0.5).
        
    Returns:
        A list of (row, col) tuples where |probability - 0.5| > target_bias.
    """
    # Calculate the bias matrix (absolute deviation from 0.5)
    bias_matrix: FloatArray = np.abs(avalanche_matrix - 0.5)
    
    # Find coordinates where bias exceeds target
    high_bias_coords: npt.NDArray[np.intp] = np.argwhere(bias_matrix > target_bias)
    
    # Convert to list of tuples
    return [(int(row), int(col)) for row, col in high_bias_coords]

if __name__ == "__main__":
    # Initialize with float64 explicit type
    curr_avalanche: FloatArray = np.eye(64, dtype=np.float64)
    x0 = 32
    curr_avalanche = apply_xor_shift(x0, curr_avalanche)
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=1, suppress=True)
    # print(curr_avalanche)
    print("x0 Constant:", x0)
    rms_bias = float(np.sqrt(np.mean((curr_avalanche - 0.5) ** 2)))
    max_bias = float(np.max(np.abs(curr_avalanche - 0.5)))
    print(f"x0 Bias: RMS={rms_bias:.6f} Max={max_bias:.6f}")

    m1: int = int(np.random.randint(0, 2**64, dtype=np.uint64))
    m1 = m1 | 1  # Ensure LSB is set to 1
    curr_avalanche = apply_multiplication_constant(m1, curr_avalanche)
    print(f"m1 Constant: {m1:016x}")
    rms_bias = float(np.sqrt(np.mean((curr_avalanche - 0.5) ** 2)))
    max_bias = float(np.max(np.abs(curr_avalanche - 0.5)))
    print(f"m1 Bias: RMS={rms_bias:.6f} Max={max_bias:.6f}")
    print(curr_avalanche)
    
    # bad_coords = find_high_bias_coordinates(curr_avalanche, 0.49)
    # print("High bias coordinates (>0.49):", bad_coords)

    x1, curr_avalanche = optimize_xor_shift(curr_avalanche, Target.MAX)
    print("x1 Constant:", x1)
    rms_bias = float(np.sqrt(np.mean((curr_avalanche - 0.5) ** 2)))
    max_bias = float(np.max(np.abs(curr_avalanche - 0.5)))
    print(f"x1 Bias: RMS={rms_bias:.6f} Max={max_bias:.6f}")

    m2, curr_avalanche = optimize_multiplication_constant2(curr_avalanche, Target.MAX)
    print(f"m2 Constant: {m2:016x}")
    rms_bias = float(np.sqrt(np.mean((curr_avalanche - 0.5) ** 2)))
    max_bias = float(np.max(np.abs(curr_avalanche - 0.5)))
    print(f"m2 Bias: RMS={rms_bias:.6f} Max={max_bias:.6f}")

    x2, curr_avalanche = optimize_xor_shift(curr_avalanche, Target.MAX)
    print("x2 Constant:", x2)
    rms_bias = float(np.sqrt(np.mean((curr_avalanche - 0.5) ** 2)))
    max_bias = float(np.max(np.abs(curr_avalanche - 0.5)))
    print(f"x2 Bias: RMS={rms_bias:.6f} Max={max_bias:.6f}")

    # Format the constants for the hillclimb64 command
    constants_str = f'[{x0} {m1:016x} {x1} {m2:016x} {x2}]'
    command = ['./hillclimb64', '-p', constants_str, '-s']
    print(' '.join(command))

    # Execute the hillclimb64 command
    result = subprocess.run(
        command,
        capture_output=False,
        text=True,
        check=True
    )
