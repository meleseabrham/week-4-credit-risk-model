# Contributing to Credit Risk Scoring Model

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** to avoid duplicates
2. **Use the issue template** (if available)
3. **Provide context**:
   - What were you trying to do?
   - What happened?
   - What did you expect?
   - Steps to reproduce
   - Environment details (OS, Python version)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Name your branch** descriptively:
   - `feat/add-woe-encoding`
   - `fix/api-timeout-error`
   - `docs/update-readme`
   - `test/add-edge-cases`

3. **Make your changes**:
   - Write clean, readable code
   - Followthe existing code style
   - Add/update tests as needed
   - Update documentation

4. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/ -v

   # Check code style
   flake8 src/ tests/ --max-line-length=88

   # Ensure Docker builds
   docker-compose build
   ```

5. **Commit your changes** using conventional commits:
   ```
   <type>(<scope>): <subject>

   <body>

   <footer>
   ```

   **Types**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation changes
   - `style`: Code style changes (formatting, no logic change)
   - `refactor`: Code refactoring
   - `test`: Adding or updating tests
   - `chore`: Maintenance tasks

   **Examples**:
   ```
   feat(api): add caching for model predictions

   Implement Redis caching to improve API response time
   for repeated predictions.

   Closes #123
   ```

6. **Push to your fork** and submit a pull request to `main`

7. **Fill out the PR template**:
   - **Problem**: What issue does this solve?
   - **Solution**: How did you solve it?
   - **Testing**: What tests did you add/run?
   - **Breaking Changes**: Are there any?
   - **Screenshots**: (if applicable)

## 📋 PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows project style guidelines (flake8 passes)
- [ ] Tests are added/updated and passing
- [ ] Documentation is updated (README, docstrings)
- [ ] Commit messages follow conventional commits format
- [ ] Branch is up-to-date with `main`
- [ ] CI checks pass
- [ ] No unnecessary files committed (.pyc, __pycache__, etc.)

## 🎨 Code Style

We follow PEP 8 with some modifications:
- **Max line length**: 88 characters (Black-compatible)
- **Imports**: Grouped and sorted (stdlib, third-party, local)
- **Docstrings**: Google-style for functions/classes
- **Type hints**: Encouraged for function signatures

### Example:

```python
def calculate_risk_score(
    transaction_amount: float,
    customer_history: pd.DataFrame
) -> float:
    """
    Calculate risk score for a transaction.

    Args:
        transaction_amount: Amount of the transaction
        customer_history: DataFrame with customer transaction history

    Returns:
        Risk score between 0 and 1

    Raises:
        ValueError: If transaction_amount is negative
    """
    if transaction_amount < 0:
        raise ValueError("Transaction amount must be positive")

    # Implementation
    return risk_score
```

## ✅ Testing Guidelines

- **Write tests for new features**
- **Update tests when modifying code**
- **Test edge cases** (empty inputs, missing values, invalid types)
- **Aim for high coverage** (but prioritize meaningful tests)

### Test Structure:

```python
def test_feature_name():
    """Test description of what is being tested."""
    # Arrange
    input_data = create_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value
```

## 📝 Documentation

- **Update README** if adding features or changing setup
- **Add docstrings** to all public functions/classes
- **Comment complex logic** but avoid obvious comments
- **Update CONTRIBUTING.md** if changing contribution process

## 🚫 What NOT to Commit

- `.env` files with secrets
- Large data files (> 10MB)
- IDE-specific files (.vscode/, .idea/)
- Compiled files (*.pyc, __pycache__)
- Model artifacts (*.pkl, *.h5) - should be in .gitignore
- Personal notes or experiments

## 🔍 Code Review Process

1. **Automated checks** must pass (CI green)
2. **At least one approving review** required
3. **Address reviewer feedback** promptly
4. **Squash and merge** to keep history clean

Reviewers will check:
- Code quality and style
- Test coverage
- Documentation completeness
- Performance implications
- Security considerations

## 🆘 Getting Help

- **Check documentation** (README, docstrings)
- **Search existing issues**
- **Ask questions** in issue comments
- **Join discussions** for design decisions

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing! 🎉
