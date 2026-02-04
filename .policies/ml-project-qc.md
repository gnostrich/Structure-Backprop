# ML Project Quality Control & Development Protocol

## Overview

This document defines a systematic process for reviewing and improving machine learning projects. It ensures coherent development across all ML repositories by establishing a repeatable cycle based on design validation, testing, and expert feedback.

## Process Overview

The QC process consists of five phases executed in sequence:

1. **Motivation Audit** - Verify implementation matches conceptual design
2. **Run Tests & Validation** - Execute code, measure performance, verify it works
3. **Export Static Outputs** - Save results in reproducible form for review
4. **Independent Expert Review** - Get external feedback on design and results
5. **Iterative Improvement** - Fix issues identified in review

---

## Phase 1: Motivation Audit

### Purpose
Ensure the implementation faithfully represents the conceptual design described in documentation.

### Checklist
- [ ] Read all design documents (README, ARCHITECTURE, design specs)
- [ ] Identify core claims about how the system works
- [ ] Locate the corresponding implementation files
- [ ] Map conceptual ideas → actual code
- [ ] Note any discrepancies between design and implementation

### Example (Continual-Learning)
```
Design Claim: "Two-model architecture with divergence-minimization learning"
Implementation Check:
  ✓ CrossModalModel exists
  ✓ EnvironmentPredictor exists
  ✓ Divergence loss computed in train_step
  ✓ Both models jointly optimized
```

### Example (Structure-Backprop)
```
Design Claim: "Dense directed graph learns structure via alternating backprop+rounding"
Implementation Check:
  ✓ Dense weight matrix initialized (n_total × n_total)
  ✓ Forward pass supports use_rounded flag
  ✓ round_weights() method exists
  ✓ train_structure_backprop alternates phases
  ? DAG constraint enforced? (NEEDS VERIFICATION)
```

---

## Phase 2: Implementation Gap Analysis

### Purpose
Systematically identify missing features, untested code paths, and incomplete implementations.

### Key Areas to Check

#### A. Testing & Validation
```python
Gaps to look for:
- No train/val/test split
- No cross-validation
- Limited dataset coverage (e.g., only XOR, Addition)
- No edge case testing
- No failure mode analysis
```

#### B. Metrics & Evaluation
```python
Gaps to look for:
- Wrong metric type (e.g., accuracy on regression)
- Missing metrics (MAE, RMSE, R², F1, etc.)
- No loss curves or convergence plots
- No comparison with baselines
- No ablation studies
```

#### C. Reproducibility
```python
Gaps to look for:
- Random seeds not fixed
- Non-deterministic operations
- Environment dependencies undocumented
- No seed reproduction guarantee
```

#### D. Constraints & Invariants
```python
Gaps to look for:
- Claims about graph structure (DAG, acyclic, etc.) not verified
- Weight bounds not enforced
- Sparsity targets not met
- Layer constraints not validated
```

#### E. Hyperparameter Tuning
```python
Gaps to look for:
- Key hyperparameters not explored (threshold, learning rate, rounding frequency)
- No sensitivity analysis
- No tuning guidance
- Default values not justified
```

#### F. Scalability
```python
Gaps to look for:
- Only tested on toy problems
- No stress testing (large graphs, many samples)
- No memory/compute profiling
- Scaling limits unknown
```

### Documentation of Gaps
Create an audit report listing:
1. **Critical gaps** (breaks conceptual design): Fix before proceeding
2. **Implementation gaps** (missing features): Add in improvement phase
3. **Validation gaps** (untested paths): Add comprehensive tests

---

## Phase 3: Validation & Testing

### Purpose
Prove the system actually works as designed and can be relied upon.

### Essential Validations

#### For All Projects
```python
✓ Run all example/demo code without errors
✓ Verify key metrics report as expected
✓ Test with multiple datasets/seeds
✓ Confirm no regressions from recent changes
✓ Check edge cases don't crash
```

#### For Structure Learning Projects
```python
✓ Verify learned graph matches expected sparsity
✓ Confirm structure is acyclic if claimed
✓ Validate weight distribution (should cluster near 0, 1)
✓ Test on tasks where optimal structure is known
```

#### For Continual Learning Projects
```python
✓ Verify learning curves are smooth
✓ Confirm catastrophic forgetting is addressed
✓ Test task switching doesn't crash system
✓ Validate divergence metrics trend correctly
```

#### For All Projects
```python
✓ Reproducibility: Same seed → same results
✓ Stability: No NaN/inf/divergence in training
✓ Generalization: Val metrics < Train metrics (if applicable)
```

### Validation Output
Document results in a file (e.g., `validate_[project].py`):
```python
"""Validation suite for [Project Name]"""

def test_basic_run():
    """Project runs without errors on demo data"""
    # Implementation
    
def test_reproducibility():
    """Same seed produces identical results"""
    # Implementation
    
def test_invariants():
    """Key properties (DAG, sparsity, bounds) hold"""
    # Implementation
    
def test_metrics():
    """Metrics match expected ranges"""
    # Implementation
```

---

## Phase 4: Iterative Improvement

### Purpose
Systematically fix gaps and enhance the implementation.

### Improvement Categories

#### Priority 1: Critical Fixes (blocks design claims)
```
Examples:
- Add DAG verification (if "directed graph" is claimed)
- Fix regression evaluation (right metric for task type)
- Implement train/val/test split (if claiming generalization)
```

#### Priority 2: Core Features (implements missing parts of design)
```
Examples:
- Add hyperparameter ablation
- Implement reproducibility (seeding)
- Add baseline comparisons
- Expand dataset coverage
```

#### Priority 3: Polish (improves usability/clarity)
```
Examples:
- Better documentation
- More example scripts
- Visualization improvements
- Code cleanup
```

### Implementation Cycle for Each Improvement
1. **Plan**: What exactly needs fixing? Why?
2. **Implement**: Make the change in Docker
3. **Test**: Verify fix works and doesn't break existing code
4. **Document**: Update README or add comment explaining change
5. **Validate**: Re-run validation suite to confirm improvement

### Example Improvement Cycle (Structure-Backprop)

```
IMPROVEMENT 1: Add DAG Verification
├─ Plan: Add is_dag() method to check for cycles
├─ Implement: BFS/topological sort based cycle detection
├─ Test: Validate on known DAG and graph with cycles
├─ Document: Explain algorithm in docstring
└─ Validate: Re-run tests, confirm all pass

IMPROVEMENT 2: Fix Regression Metrics
├─ Plan: Add MAE, RMSE, R² for regression tasks
├─ Implement: Update example.py to use right metrics
├─ Test: Verify metrics are computed correctly
├─ Document: Update README with metric definitions
└─ Validate: Confirm Addition problem now uses correct eval

IMPROVEMENT 3: Add Train/Val/Test Split
├─ Plan: Implement train_val_test_split() function
├─ Implement: Split with stratification if classification
├─ Test: Verify splits are non-overlapping and sized correctly
├─ Document: Add usage example to README
└─ Validate: Verify val loss > train loss pattern
```

---

## Integration with Development Workflow

### Scope Boundaries
- **Policy Level** (this document + referenced standards): Host-level
- **Project Level** (code, commits, validation): Docker-only

### When to Apply QC Process
1. **New project onboarding**: Full audit (all 4 phases)
2. **Major feature addition**: Phases 2-4 (revalidate)
3. **Bug fix**: Phase 3 (regression testing)
4. **Routine checkup**: Quick Phase 1 (design still matches?)

### Checklist: QC Complete?
- [ ] Phase 1: Motivation audit document written
- [ ] Phase 2: Gap analysis report created
- [ ] Phase 3: Validation suite runs and passes
- [ ] Phase 4: Improvements planned and tracked
- [ ] All changes committed with descriptive messages
- [ ] README updated with findings
- [ ] Docker container used for all code work

---

## Template: QC Report

Use this template for documenting QC results:

```markdown
# [Project] QC Report - [Date]

## Phase 1: Motivation Audit

**Conceptual Design Claims:**
- Claim 1: [describe]
  - Implementation: [✓/✗] [details]
- Claim 2: [describe]
  - Implementation: [✓/✗] [details]

## Phase 2: Gap Analysis

**Critical Gaps (blocks design):**
1. [Gap description] - MUST FIX
2. ...

**Implementation Gaps (missing features):**
1. [Gap description] - NICE TO HAVE
2. ...

**Validation Gaps (untested):**
1. [Gap description] - SHOULD TEST
2. ...

## Phase 3: Validation Results

- [x] Basic runs without errors
- [x] Reproducibility: [PASS/FAIL]
- [ ] Invariants: [PASS/FAIL]
- [ ] Metrics: [PASS/FAIL]

## Phase 4: Improvements Planned

1. **Fix DAG verification** (Priority 1)
   - Status: [NOT STARTED/IN PROGRESS/DONE]
   - PR: [link]

2. **Add regression metrics** (Priority 1)
   - Status: [NOT STARTED/IN PROGRESS/DONE]
   - PR: [link]

3. [Additional improvements...]
```

---

## References

**Related Policies:**
- Docker-First Development Protocol (docker-first.md)
- Autonomous Git & PR Policy (autonomous-git.md)

**Tools & Libraries:**
- PyTorch testing: `torch.manual_seed()`, `torch.testing.assert_close()`
- Data splitting: `sklearn.model_selection.train_test_split`
- Metrics: `sklearn.metrics`, `torchmetrics`

**ML Best Practices:**
- Always use train/val/test splits
- Set random seeds for reproducibility
- Test on multiple datasets
- Compare against baselines
- Ablate hyperparameters systematically
