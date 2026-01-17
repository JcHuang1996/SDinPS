# SDinPS - Stochastic Distribution in Power Systems

This project implements mixed-integer linear programming (MILP) models for power system optimization using decomposition algorithms.

## Quick Start

Run tests from the project root directory:

```bash
python main.py <test_name> [options]
```

## Available Tests

The following tests are available:

- **`combined_formulation`**: Run the combined formulation test
- **`decomposition_algo`**: Run the decomposition algorithm module test

## Running Tests

### Combined Formulation Test

Run the combined formulation test with default settings:

```bash
python main.py combined_formulation
```

### Decomposition Algorithm Test

The decomposition algorithm test supports multiple command-line parameters. All parameters are optional and will use default values if not specified.

#### Basic Usage

Run with all default parameters:

```bash
python main.py decomposition_algo
```

#### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable-log-output` | flag | `False` | Enable writing running logs to log output folder |
| `--disable-result-output` | flag | `True` | Disable result output (default: results are saved) |
| `--scenario-list` | string | `s_1,s_2,s_3` | Comma-separated list of scenarios to test |
| `--time-list` | string | `1,2,3,4,5,6,7,8,9,10,11` | Comma-separated list of time periods |
| `--test-file-path` | string | `/Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file` | Path to test data files |
| `--data-set-name` | string | `function test` | Name of the data set |
| `--max-iterations` | integer | `10` | Maximum number of iterations to run |

#### Sample Command Lines

**Example 1: Run with default settings**
```bash
python main.py decomposition_algo
```

**Example 2: Enable log output**
```bash
python main.py decomposition_algo --enable-log-output
```

**Example 3: Custom scenario list**
```bash
python main.py decomposition_algo --scenario-list s_1,s_2,s_3,s_4
```

**Example 4: Custom time periods**
```bash
python main.py decomposition_algo --time-list 1,2,3,4,5
```

**Example 5: Custom scenarios and time periods**
```bash
python main.py decomposition_algo --scenario-list s_1,s_2,s_3 --time-list 1,2,3,4,5,6,7,8
```

**Example 6: Custom file path**
```bash
python main.py decomposition_algo --test-file-path /path/to/your/test/data
```

**Example 7: Custom data set name**
```bash
python main.py decomposition_algo --data-set-name "my custom test"
```

**Example 8: Disable result output**
```bash
python main.py decomposition_algo --disable-result-output
```

**Example 9: Custom number of iterations**
```bash
python main.py decomposition_algo --max-iterations 20
```

**Example 10: Combine multiple options**
```bash
python main.py decomposition_algo --enable-log-output --scenario-list s_1,s_2 --time-list 1,2,3 --data-set-name "quick test"
```

**Example 11: Full custom configuration**
```bash
python main.py decomposition_algo \
  --enable-log-output \
  --scenario-list s_1,s_2,s_3,s_4,s_5 \
  --time-list 1,2,3,4,5,6,7,8,9,10 \
  --test-file-path /Users/huangjiacheng/SDinPS/unit_test/test_local_csv_file \
  --data-set-name "full test" \
  --max-iterations 15
```

## Notes

- All parameters for `decomposition_algo` are optional. If not specified, default values will be used.
- Scenario lists should be comma-separated without spaces (or with spaces, they will be trimmed automatically).
- Time lists should be comma-separated integers.
- The test will create output directories with timestamps in the `output/` folder when result output is enabled.
- Log files will be written to the `logs/` folder when log output is enabled.

## Getting Help

To see available options for a specific test, use:

```bash
python main.py --help
```

This will display all available command-line arguments and their descriptions.

