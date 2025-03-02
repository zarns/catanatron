# Catanatron Examples

This directory contains example scripts that demonstrate how to use various features of the Catanatron project.

## Available Examples

### Rust Backend Usage

- **use_rust_backend.py**: Demonstrates how to use the Rust backend directly, comparing performance between Python and Rust implementations.

  ```bash
  # Run with default number of games (10)
  python examples/use_rust_backend.py
  
  # Run with custom number of games
  python examples/use_rust_backend.py 100
  ```

## Running Examples

Most examples can be run directly from the project root:

```bash
python examples/example_script.py
```

Make sure you have installed the project dependencies as described in the main README.md file.

## Creating Your Own Examples

To create your own examples, follow these guidelines:

1. Create a new Python file in the examples directory.
2. Start with a clear docstring explaining the purpose of the example.
3. Use relative imports to import from the Catanatron packages.
4. Include comments to explain key steps in your code.
5. Add your example to this README.md file.

If your example demonstrates a complex feature or technique, consider writing a more detailed explanation in a comment at the top of the file. 