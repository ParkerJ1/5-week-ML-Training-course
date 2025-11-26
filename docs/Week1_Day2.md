# Week 1, Day 2: Python Crash Course - Matplotlib & Data Visualization

## Daily Goals

- Master basic plotting with Matplotlib
- Create various plot types (line, scatter, histogram, heatmap)
- Understand when to use different visualizations

---

## Morning Session (4 hours)

### Optional: Daily Check-in with Peers on Teams (15 min)

### Video Learning (90 min)

☐ **Watch**: [Matplotlib Tutorial for Beginners](https://www.youtube.com/watch?v=3Xc3CA655Y4) by Corey Schafer (1 hour)

☐ **Watch**: [Matplotlib Crash Course](https://www.youtube.com/watch?v=DAQNHzOcO5A) by Tech With Tim - focus on 0:00-30:00 (30 min)

### Reference Material (30 min)

☐ **Read**: [Matplotlib Quick Start Guide](https://matplotlib.org/stable/tutorials/introductory/quick_start.html)

☐ **Bookmark**: [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) for examples

### Hands-on Coding - Part 1 (2 hours)

#### Setup (10 min)

☐ Create a new Colab notebook titled "Day2_Matplotlib_Practice"

☐ Import libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
```

#### Exercise 1: Basic Line Plots (40 min)

**1. Simple line plot**: Plot y = x² for x values from -10 to 10

```python
x = np.linspace(-10, 10, 100)
# Your code here
```

*Hint: Use `plt.plot()` then `plt.show()`*
*Expected: A parabola centered at origin*

**2. Multiple lines**: Plot sin(x), cos(x), and tan(x) for x from 0 to 2π on the same graph

```python
x = np.linspace(0, 2*np.pi, 100)
# Your code here
# Add labels, legend, title, and grid
```

*Hint: Call `plt.plot()` multiple times before `plt.show()`*
*Use `plt.legend()`, `plt.xlabel()`, `plt.ylabel()`, `plt.title()`, `plt.grid()`*

**3. Customized plot**: Create a line plot with:

- Custom colors (use 'red', 'blue', etc.)
- Different line styles (solid, dashed, dotted)
- Markers at data points
- Custom line widths

```python
x = np.linspace(0, 10, 20)
y1 = x
y2 = x**2
# Your code here
```

*Hint: Use parameters like `color='red'`, `linestyle='--'`, `marker='o'`, `linewidth=2`*

#### Exercise 2: Scatter Plots (35 min)

**1. Basic scatter plot**: Create random data and visualize

```python
np.random.seed(42)
x = np.random.randn(100)
y = 2*x + np.random.randn(100)*0.5
# Create scatter plot
# Add title and labels
```

*Hint: Use `plt.scatter()`*
*Expected: Points roughly following a line with some noise*

**2. Colored scatter plot**: Create a scatter plot where point colors represent a third variable

```python
x = np.random.rand(50)
y = np.random.rand(50)
colors = x + y  # Color based on sum of coordinates
sizes = (x * 100) ** 2  # Size based on x value
# Create scatter plot with colors and varying sizes
# Add a colorbar
```

*Hint: Use `c=colors`, `s=sizes` parameters in `plt.scatter()`, then `plt.colorbar()`*

#### Exercise 3: Subplots (35 min)

Create a 2x2 grid of subplots showing different mathematical functions:

```python
x = np.linspace(-5, 5, 100)

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Top-left: y = x²
# Top-right: y = e^x
# Bottom-left: y = sin(x)
# Bottom-right: y = e^x * sin(x)

# Add titles to each subplot
# Add a main title to the figure
```

*Hint: Access subplots using `axes[row, col].plot()`*
*Use `axes[row, col].set_title()` for individual titles*
*Use `fig.suptitle()` for main title*
*Expected: Four distinct plots in a grid layout*

---

## Afternoon Session (4 hours)

### Video Learning (30 min)

☐ **Watch**: [Matplotlib Histograms and Bar Charts](https://www.youtube.com/watch?v=XDv6T4a0RNc) by Corey Schafer (15 min)

☐ **Watch**: [Seaborn Heatmaps](https://www.youtube.com/watch?v=6GUZXDef2U0) first 15 minutes (15 min)

### Hands-on Coding - Part 2 (3 hours)

#### Exercise 4: Histograms (40 min)

**1. Single histogram**: Generate and visualize normal distribution

```python
data = np.random.randn(1000)
# Create histogram with 30 bins
# Add labels and title
```

*Hint: Use `plt.hist(data, bins=30)`*
*Expected: Bell curve shape*

**2. Overlapping histograms**: Compare two distributions

```python
data1 = np.random.randn(1000)
data2 = np.random.randn(1000) + 2  # Shifted distribution
# Plot both histograms with transparency
# Add legend
```

*Hint: Use `alpha=0.5` for transparency*
*Use different colors for each histogram*

**3. Histogram analysis**: Generate data and analyze with histogram

```python
# Generate data from mixed distributions
data = np.concatenate([
    np.random.randn(500),
    np.random.randn(500) + 5
])
# Create histogram
# What pattern do you observe?
```

*Expected: Bimodal distribution (two peaks)*

#### Exercise 5: Heatmaps (45 min)

**1. Correlation matrix**: Create and visualize a correlation matrix

```python
# Generate correlated data
np.random.seed(42)
data = np.random.randn(100, 5)
# Calculate correlation matrix
corr_matrix = np.corrcoef(data.T)

# Create heatmap
# Add colorbar and labels
```

*Hint: Use `plt.imshow()` with `cmap='coolwarm'`*
*Use `plt.colorbar()` to show scale*

**2. 2D function visualization**: Visualize z = sin(x) * cos(y)

```python
x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create heatmap
# Add colorbar, labels, and title
```

*Hint: Use `plt.imshow()` or `plt.contourf()`*
*Expected: Symmetric pattern with positive and negative regions*

#### Exercise 6: Bar Charts (35 min)

**1. Simple bar chart**: Visualize category data

```python
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 45, 56, 78]
# Create bar chart
# Rotate x-labels if needed
```

*Hint: Use `plt.bar()`*
*Use `plt.xticks(rotation=45)` if labels overlap*

**2. Grouped bar chart**: Compare multiple series

```python
categories = ['Q1', 'Q2', 'Q3', 'Q4']
product_A = [50, 60, 70, 80]
product_B = [45, 55, 65, 85]

# Create grouped bar chart showing both products
# Add legend
```

*Hint: Use `x` positions and offset bars with width parameter*
*Example: `plt.bar(x - width/2, product_A)` and `plt.bar(x + width/2, product_B)`*

#### Mini-Challenge: Data Visualization Dashboard (60 min)

Create a comprehensive visualization dashboard for a synthetic dataset:

```python
# Generate synthetic student performance data
np.random.seed(42)
n_students = 200

study_hours = np.random.uniform(0, 10, n_students)
attendance = np.random.uniform(50, 100, n_students)
exam_scores = (study_hours * 5 + attendance * 0.3 + 
               np.random.randn(n_students) * 5)
exam_scores = np.clip(exam_scores, 0, 100)
```

Create a figure with 6 subplots (2 rows × 3 columns) showing:

1. **Scatter plot**: Study hours vs exam scores (with trendline)
   *Hint: Use `np.polyfit()` and `np.poly1d()` for trendline*

2. **Scatter plot**: Attendance vs exam scores (colored by study hours)
   *Hint: Use `c=study_hours` and add colorbar*

3. **Histogram**: Distribution of exam scores
   
   - Mark mean and median with vertical lines
     *Hint: Use `plt.axvline()` for vertical lines*

4. **Histogram**: Distribution of study hours

5. **Box plot**: Exam scores grouped by study hour ranges (0-3, 3-6, 6-10 hours)
   
   ```python
   # Categorize students
   low_study = exam_scores[study_hours < 3]
   med_study = exam_scores[(study_hours >= 3) & (study_hours < 6)]
   high_study = exam_scores[study_hours >= 6]
   ```
   
   *Hint: Use `plt.boxplot([low_study, med_study, high_study])`*

6. **Heatmap**: 2D histogram (study hours vs attendance), colored by average exam score
   *Hint: Use `plt.hist2d()` or bin the data manually*

**Requirements**:

- All subplots should have appropriate titles, labels, and legends where needed
- Use a consistent color scheme
- Add a main title to the entire figure
- Make sure the figure is large enough to see all details clearly

---

## Reflection & Consolidation (30 min)

☐ Review all visualizations you created today

☐ Experiment with different color schemes and styles

☐ Write your daily reflection (choose 2-3 prompts below)

☐ Note any questions for the check-in email

### Daily Reflection Prompts (Choose 2-3):

- What was the most important concept you learned today?
- What challenged you the most? How did you approach it?
- What connections did you make between yesterday's NumPy work and today's visualization?
- What questions do you still have?
- How could these visualization techniques be useful in understanding data?
- What would you do differently if you repeated today?

---

**Next**: [Day 3 - Introduction to Machine Learning](Week1_Day3.md)
