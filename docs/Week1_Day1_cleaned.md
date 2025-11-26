# Week 1, Day 1: Python Crash Course - NumPy & Arrays

## Daily Goals

- Set up Python environment (Anaconda/Colab)
- Master NumPy array operations
- Understand vectorization benefits

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

- [ ] **Watch**: [NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI) by freeCodeCamp (1 hour)

- [ ] **Watch**: [Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=8Y0qQEh7dJg) by Keith Galli - focus on 0:00-15:00 for basics and 30:00-45:00 for array operations (30 min)

### Reference Material (30 min)

- [ ] **Read**: [NumPy Quickstart Tutorial](https://numpy.org/doc/stable/user/quickstart.html) - Sections on "The Basics" and "Array Creation"

- [ ] **Bookmark**: [NumPy Reference Documentation](https://numpy.org/doc/stable/reference/index.html) for afternoon use

### Hands-on Coding - Part 1 (2 hours)

#### Setup (15 min)

- [ ] Create a Google Colab notebook titled "Day1_NumPy_Practice"

- [ ] Import NumPy: `import numpy as np`

- [ ] Test installation: `print(np.__version__)`

#### Exercise 1: Array Creation (30 min)

Create the following arrays and print their shape, dtype, and contents:

1. A 1D array with integers from 0 to 9
   
   - *Hint: Use `np.arange()`*

2. A 2D array (3x3) filled with zeros
   
   - *Hint: Use `np.zeros()`*

3. A 3x3 identity matrix
   
   - *Hint: Use `np.eye()`*

4. A 1D array with 10 evenly spaced values between 0 and 1
   
   - *Hint: Use `np.linspace()`*

5. A 2D array (5x5) with random values between 0 and 1
   
   - *Hint: Use `np.random.rand()`*

6. A 2D array (3x4) with sequential values from 0 to 11
   
   - *Hint: Use `np.arange()` then `.reshape()`*

#### Exercise 2: Array Indexing and Slicing (40 min)

Given this array:

```python
arr = np.arange(0, 100).reshape(10, 10)
```

Complete these tasks:

1. Extract the element at row 3, column 5
   
   - *Expected output: 35*

2. Extract the entire 5th row - remember Python starts at index 0
   
   - *Expected output: array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59])*

3. Extract the top-left 3x3 subarray
   
   - *Hint: Use slicing `arr[start:end, start:end]`*

4. Extract every other row
   
   - *Hint: Use step in slicing `arr[::2]`*

5. Extract the bottom-right 2x2 subarray
   
   - *Hint: Use negative indices*

6. Extract all elements greater than 50 (returns 1D array)
   
   - *Hint: Use boolean indexing `arr[arr > 50]`*

#### Exercise 3: Basic Array Operations (35 min)

Create two arrays:

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
```

Perform and print results:

1. Element-wise addition (a + b)
   
   - *Expected output: [11, 22, 33, 44, 55]*

2. Element-wise multiplication (a * b)
   
   - *Expected output: [10, 40, 90, 160, 250]*

3. Square each element of array a
   
   - *Hint: Use `a ** 2`*

4. Calculate the dot product of a and b
   
   - *Hint: Use `np.dot()` or `@` operator*
   - *Expected output: 550*

5. Find the sum, mean, and standard deviation of array a
   
   - *Hint: Use `np.sum()`, `np.mean()`, `np.std()`*

6. Find the maximum value in b and its index
   
   - *Hint: Use `np.max()` and `np.argmax()`*

---

## Afternoon Session (4 hours)

### Video Learning (30 min)

- [ ] **Watch**: [NumPy Broadcasting](https://www.youtube.com/watch?v=oG1t3qlzq14) by Algorithmic Simplicity (15 min)

- [ ] **Watch**: [Vectorization in Python](https://www.youtube.com/watch?v=qS1HJW-x2wY) first half (15 min)

### Hands-on Coding - Part 2 (3 hours)

#### Exercise 4: Statistics Without Loops (50 min)

Implement these functions using only NumPy operations (no Python loops):

**1. normalize_array(arr)**: Takes a 1D array and returns it normalized to have mean=0 and std=1

```python
def normalize_array(arr):
    # Your code here
    pass

# Test
arr = np.array([1, 2, 3, 4, 5])
result = normalize_array(arr)
print(f"Mean: {result.mean():.10f}, Std: {result.std():.10f}")
# Expected: Mean ≈ 0, Std ≈ 1
```

*Hint: normalized = (arr - mean) / std*

**2. moving_average(arr, window_size)**: Calculate moving average with given window size

```python
def moving_average(arr, window_size):
    # Your code here
    pass

# Test
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
result = moving_average(arr, 3)
print(result)
# Expected output: [2., 3., 4., 5., 6., 7., 8., 9.]
```

*Hint: Use `np.convolve()` with `mode='valid'`*

**3. find_outliers(arr, threshold)**: Return indices where values deviate from the mean by more than threshold * std

```python
def find_outliers(arr, threshold):
    # Your code here
    pass

# Test
arr = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8])
outliers = find_outliers(arr, 2)
print(f"Outlier indices: {outliers}")
print(f"Outlier values: {arr[outliers]}")
# Expected: Index 5 (value 100) is an outlier
```

*Hint: Calculate z-scores: `(arr - mean) / std`, then find where `abs(z_scores) > threshold`*

#### Exercise 5: Broadcasting Practice (40 min)

Solve these without loops:

1. Add a 1D array `[1, 2, 3]` to each row of a 4x3 matrix
   
   ```python
   matrix = np.array([[1, 1, 1],
                   [2, 2, 2],
                   [3, 3, 3],
                   [4, 4, 4]])
   to_add = np.array([1, 2, 3])
   # Your code here
   # Expected output:
   # [[2, 3, 4],
   #  [3, 4, 5],
   #  [4, 5, 6],
   #  [5, 6, 7]]
   ```
   
   *Hint: Broadcasting works automatically with `matrix + to_add`*

2. Create a 10x10 multiplication table using broadcasting
   
   ```python
   # Your code here
   # Expected: Element (i,j) should be i*j
   # Row 5 should be [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
   ```
   
   *Hint: Create two arrays with shapes (10, 1) and (1, 10), then multiply*

3. Normalize each column of a matrix independently to range [0, 1]
   
   ```python
   matrix = np.random.rand(5, 3) * 100
   # Your code here: normalize each column so min=0, max=1
   # Verify: Check that each column's min is 0 and max is 1
   ```
   
   *Hint: For each column: `(col - col.min()) / (col.max() - col.min())`*
   *Use broadcasting with `axis=0` in min/max functions*

#### Mini-Challenge: Image Manipulation (90 min)

You'll work with a grayscale image represented as a 2D NumPy array. First, create a test image:

```python
# Create a simple test pattern
image = np.zeros((100, 100))
image[25:75, 25:75] = 1.0  # White square in center
image[40:60, 40:60] = 0.5  # Gray square in middle

# Visualize (you'll need this throughout)
import matplotlib.pyplot as plt
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.show()
```

Implement these image transformations:

**1. rotate_90(image)**: Rotate image 90 degrees clockwise

```python
def rotate_90(image):
    # Your code here
    pass

result = rotate_90(image)
plt.imshow(result, cmap='gray')
plt.title('Rotated 90° clockwise')
plt.show()
```

*Hint: Use `np.rot90()` with appropriate k value*

**2. flip_horizontal(image)**: Flip image horizontally (mirror left-right)

```python
def flip_horizontal(image):
    # Your code here
    pass

result = flip_horizontal(image)
plt.imshow(result, cmap='gray')
plt.title('Flipped Horizontally')
plt.show()
```

*Hint: Use `np.fliplr()` or slicing with `::-1`*

**3. crop_center(image, size)**: Crop a square of given size from the center

```python
def crop_center(image, size):
    # Your code here
    # Example: if image is 100x100 and size=50, extract the central 50x50 region
    pass

result = crop_center(image, 50)
print(f"Cropped shape: {result.shape}")  # Should be (50, 50)
plt.imshow(result, cmap='gray')
plt.title('Center Crop (50x50)')
plt.show()
```

*Hint: Calculate center indices, then slice appropriately*

**4. adjust_brightness(image, factor)**: Multiply all pixels by factor and clip to [0, 1]

```python
def adjust_brightness(image, factor):
    # Your code here
    pass

# Test with darkening (factor=0.5) and brightening (factor=1.5)
dark = adjust_brightness(image, 0.5)
bright = adjust_brightness(image, 1.5)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(dark, cmap='gray')
axes[1].set_title('Darkened (0.5x)')
axes[2].imshow(bright, cmap='gray')
axes[2].set_title('Brightened (1.5x)')
plt.show()
```

*Hint: Use `np.clip(image * factor, 0, 1)`*

**5. apply_threshold(image, threshold)**: Convert to binary (0 or 1) based on threshold

```python
def apply_threshold(image, threshold):
    # Your code here
    # Pixels >= threshold become 1, others become 0
    pass

result = apply_threshold(image, 0.5)
plt.imshow(result, cmap='gray')
plt.title('Thresholded at 0.5')
plt.show()
```

*Hint: Use boolean comparison and `.astype(float)`*

---

## Reflection & Consolidation (30 min)

- [ ] Review all code you wrote today

- [ ] Clean up your notebook with comments and markdown cells

- [ ] Write your daily reflection in a separate document (choose 2-3 prompts below)

- [ ] List any questions for the Monday check-in email

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- What challenged you the most? How did you approach it?
- What connections did you make between today's content and previous learning?
- What questions do you still have?
- How does today's learning relate to real-world applications?
- What would you do differently if you repeated today?

---

**Next**: [Day 2 - Matplotlib & Data Visualization](Week1_Day2.md)
