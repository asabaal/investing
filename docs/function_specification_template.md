# Function Specification Template

## Overview

| Field | Description |
|-------|-------------|
| **Function Name** | Name of the function |
| **Module** | File/module where function is located |
| **Purpose** | Clear one-sentence description of what the function does |
| **Current Status** | Brief assessment of current implementation state |

## Specification

### Inputs
- List parameter names, types, and descriptions
- Include whether parameters are optional or required
- Document any constraints on parameter values

### Outputs
- Describe the return value(s) and their type(s)
- Document the expected format of complex return values
- List potential exceptions/errors that can be raised

### Behavior
- Detailed description of what the function does
- Step-by-step description of the algorithm or process
- Any side effects (e.g., modifying global state, writing to files)

## Implementation Assessment

### Current Implementation Issues
- List problems with current implementation (complexity, length, etc.)
- Identify any potential bugs or edge cases not handled
- Note areas where documentation is unclear or missing

### Single Responsibility Analysis
- Is this function doing too many things?
- Suggested decomposition into smaller functions

### Dependencies
- List functions/modules this function depends on
- Identify any problematic tight coupling

## Testing Strategy

### Test Cases
- List specific inputs and expected outputs to verify correctness
- Include edge cases and error conditions

### Mocking Requirements
- What external dependencies should be mocked for testing

## Refactoring Plan

### Proposed Changes
- Detailed list of specific changes to make
- How the function will be decomposed (if needed)

### New Function Signatures
- Define signatures for any new functions to be created

### Backward Compatibility
- How will changes maintain compatibility with existing code

## Documentation

### Function Docstring
- Proposed docstring for the refactored function

### Usage Example
- Code snippet showing how to use the function correctly

## Implementation Checklist

- [ ] Write tests for current behavior
- [ ] Implement refactored function(s)
- [ ] Update documentation
- [ ] Verify tests pass
- [ ] Verify backward compatibility
