# Logging Utils Documentation

## Overview

The `logging_utils.py` module provides a centralized logging system with hierarchical query ID management for the SQL Vector Retrieval application. It enables tracking of main queries and their sub-queries through unique identifiers that reset daily and are thread-safe for concurrent processing.

## Key Features

- **Hierarchical Query IDs**: Main queries (1, 2, 3...) and sub-queries (1.1, 1.2, 2.1, 2.2...)
- **Thread-Safe Operations**: Supports concurrent request processing
- **Daily Reset**: Query counters automatically reset each day
- **Context Management**: Automatic query ID propagation through function calls
- **File-Only Logging**: Logs written to files without console output

## Architecture

### Core Components

1. **QueryIDManager**: Manages query ID generation and daily reset
2. **QueryIDFormatter**: Custom log formatter that includes query IDs
3. **Context Managers**: Handle query ID context for main and sub-queries
4. **Context Variables**: Thread-safe storage for query IDs

## Detailed Component Documentation

### 1. QueryIDManager Class

```python
class QueryIDManager:
    """Thread-safe query ID manager that resets daily and supports sub-queries"""
```

#### Purpose
Manages the generation of unique query IDs for both main queries and sub-queries, with automatic daily reset functionality.

#### Key Methods

##### `get_next_main_id() -> int`
- **Purpose**: Generates the next unique main query ID
- **Returns**: Sequential integer (1, 2, 3, 4...)
- **Thread-Safe**: Yes, uses internal lock
- **Daily Reset**: Automatically resets to 1 each day

**Example:**
```python
query_id_manager = QueryIDManager()
main_id = query_id_manager.get_next_main_id()  # Returns: 1, 2, 3...
```

##### `get_next_sub_id(main_id: int) -> str`
- **Purpose**: Generates the next unique sub-query ID for a given main query
- **Parameters**: `main_id` - The main query ID to create sub-query for
- **Returns**: String in format "main_id.sub_number" (e.g., "2.1", "2.2")
- **Thread-Safe**: Yes, uses internal lock
- **Daily Reset**: Sub-counters reset daily along with main counter

**Example:**
```python
main_id = 2
sub_id_1 = query_id_manager.get_next_sub_id(main_id)  # Returns: "2.1"
sub_id_2 = query_id_manager.get_next_sub_id(main_id)  # Returns: "2.2"
```

##### `get_current_date() -> str`
- **Purpose**: Gets the current date string for log file naming
- **Returns**: Date in format "YYYY-MM-DD"

#### Internal State Management

```python
self._main_counter = 0           # Counter for main query IDs
self._sub_counters = {}          # Dict: {main_id: sub_counter}
self._current_date = None        # Current date for reset logic
self._lock = threading.Lock()    # Thread safety lock
```

### 2. QueryIDFormatter Class

```python
class QueryIDFormatter(logging.Formatter):
    """Custom formatter that includes query ID in log messages"""
```

#### Purpose
Extends Python's standard logging formatter to automatically include query IDs in log messages.

#### Log Format
```
%(asctime)s - %(name)s - %(levelname)s - [Query ID:%(query_id)s] - %(message)s
```

**Example Output:**
```
2025-10-15 12:52:02,324 - api_main - INFO - [Query ID:2] - Original received query: What is India's GDP?
2025-10-15 12:52:02,326 - handler_sql - INFO - [Query ID:2.1] - START: Processing query: GDP data for India
```

#### How It Works
1. Retrieves query ID from context variables
2. Sets the `query_id` field on the log record
3. Formats the message with the query ID included

### 3. Context Management

#### Context Variables
```python
query_id_context = contextvars.ContextVar('query_id', default=None)
main_query_id_context = contextvars.ContextVar('main_query_id', default=None)
```

- **`query_id_context`**: Stores the current query ID (main or sub-query)
- **`main_query_id_context`**: Stores the main query ID for reference

#### QueryIDContext Class

```python
class QueryIDContext:
    """Context manager for main query ID"""
```

**Purpose**: Manages context for main query processing.

**Usage:**
```python
with QueryIDContext(main_query_id=2):
    logger.info("Processing main query")  # Logs with [Query ID:2]
    # All logging within this context uses main query ID
```

**What it does:**
1. Sets both `query_id_context` and `main_query_id_context` to the main query ID
2. Automatically restores previous context when exiting

#### SubQueryIDContext Class

```python
class SubQueryIDContext:
    """Context manager for sub-query ID"""
```

**Purpose**: Manages context for sub-query processing.

**Usage:**
```python
with SubQueryIDContext(sub_query_id="2.1", main_query_id=2):
    logger.info("Processing sub-query")  # Logs with [Query ID:2.1]
    # All logging within this context uses sub-query ID
```

**What it does:**
1. Sets `query_id_context` to the sub-query ID
2. Sets `main_query_id_context` to the main query ID for reference
3. Automatically restores previous context when exiting

### 4. Setup and Configuration Functions

#### `setup_logging(filename_prefix: str, level: int = logging.INFO) -> logging.Logger`

**Purpose**: Configures the logging system with query ID support.

**Parameters:**
- `filename_prefix`: Prefix for log filename (e.g., 'sql', 'api', 'metadata')
- `level`: Logging level (default: INFO)

**What it does:**
1. Creates log filename: `{prefix}-{date}.log`
2. Sets up file handler with QueryIDFormatter
3. Configures root logger
4. Removes console output (file-only logging)

**Example:**
```python
setup_logging("sql", logging.INFO)  # Creates: sql-2025-10-15.log
```

#### `get_logger(name: str) -> logging.Logger`

**Purpose**: Gets a logger instance with query ID support.

**Usage:**
```python
logger = get_logger(__name__)
logger.info("This message will include query ID automatically")
```

### 5. Utility Functions

#### `get_query_id() -> Optional[Union[int, str]]`
- **Purpose**: Gets the current query ID from context
- **Returns**: Main query ID (int) or sub-query ID (str), or None

#### `get_main_query_id() -> Optional[int]`
- **Purpose**: Gets the current main query ID from context
- **Returns**: Main query ID (int) or None

#### `clear_query_id() -> None`
- **Purpose**: Clears query ID context
- **Usage**: Typically not needed as context managers handle cleanup

## Usage Examples

### Basic Usage

```python
from logging_utils import setup_logging, get_logger, query_id_manager, QueryIDContext, SubQueryIDContext

# Setup logging
setup_logging("api", logging.INFO)
logger = get_logger(__name__)

# Main query processing
main_query_id = query_id_manager.get_next_main_id()  # Returns: 1

with QueryIDContext(main_query_id):
    logger.info("Starting main query")  # Logs: [Query ID:1] Starting main query
    
    # Process sub-queries
    sub_query_id = query_id_manager.get_next_sub_id(main_query_id)  # Returns: "1.1"
    
    with SubQueryIDContext(sub_query_id, main_query_id):
        logger.info("Processing sub-query")  # Logs: [Query ID:1.1] Processing sub-query
```

### Concurrent Processing

```python
import threading

def process_query(query_text):
    main_query_id = query_id_manager.get_next_main_id()
    
    with QueryIDContext(main_query_id):
        logger.info(f"Processing: {query_text}")
        
        # Multiple sub-queries
        for i in range(2):
            sub_id = query_id_manager.get_next_sub_id(main_query_id)
            with SubQueryIDContext(sub_id, main_query_id):
                logger.info(f"Sub-query {i+1}")

# Concurrent execution
threads = []
for query in ["Query 1", "Query 2", "Query 3"]:
    thread = threading.Thread(target=process_query, args=(query,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

## Integration with Application

### API Main Flow

```python
# api_main.py
@app.post("/integrated_query")
async def orchestrate(question: Question):
    main_query_id = query_id_manager.get_next_main_id()
    
    with QueryIDContext(main_query_id):
        logger.info("Original received query: " + question.question)
        
        # Generate sub-queries
        sql_queries, _, _ = generate_sql_queries(rephrased_query)
        
        # Process sub-queries
        sql_responses = await batch_sql_queries(batch, user_query, total_input_tokens, total_output_tokens, main_query_id)
```

### Handler SQL Flow

```python
# handler_sql.py
async def batch_sql_queries(batch, orig_query, total_input_tokens, total_output_tokens, main_query_id):
    async def limited(q, orig_query, sub_query_index):
        sub_query_id = query_id_manager.get_next_sub_id(main_query_id)
        
        with SubQueryIDContext(sub_query_id, main_query_id):
            return await process_single_query(q, orig_query, nq, total_input_tokens, total_output_tokens)
```

## Log File Structure

### File Naming
- Format: `{prefix}-{YYYY-MM-DD}.log`
- Examples: `sql-2025-10-15.log`, `api-2025-10-15.log`

### Log Entry Format
```
2025-10-15 12:52:02,324 - module_name - LEVEL - [Query ID:X] - message
```

### Example Log Entries
```
2025-10-15 12:52:02,324 - api_main - INFO - [Query ID:2] - Original received query: What is India's GDP and inflation?
2025-10-15 12:52:02,326 - handler_sql - INFO - [Query ID:2.1] - START: Processing query: GDP data for India
2025-10-15 12:52:02,329 - handler_sql - INFO - [Query ID:2.2] - START: Processing query: Inflation data for India
2025-10-15 12:52:02,330 - api_main - INFO - [Query ID:2] - Main query processing completed
```

## Thread Safety

The module is fully thread-safe through:

1. **Threading.Lock**: Protects query ID generation
2. **Context Variables**: Thread-local storage for query IDs
3. **Context Managers**: Automatic cleanup and isolation

## Daily Reset Mechanism

- **Trigger**: Automatically checks date on each ID generation
- **Reset Time**: At midnight (when date changes)
- **What Resets**: Both main counter and all sub-counters
- **Persistence**: No persistence across application restarts

## Error Handling

- **Missing Context**: Logs show "N/A" for query ID if context is not set
- **Context Cleanup**: Automatic cleanup via context managers
- **Thread Safety**: Lock prevents race conditions

## Performance Considerations

- **Minimal Overhead**: Context variables are lightweight
- **Lock Contention**: Minimal due to fast ID generation
- **Memory Usage**: Sub-counters dictionary grows with unique main query IDs per day

## Best Practices

1. **Always use context managers** for query ID management
2. **Don't manually set query IDs** unless absolutely necessary
3. **Use descriptive logger names** for better log filtering
4. **Monitor log file sizes** for long-running applications
5. **Use appropriate log levels** (INFO, DEBUG, ERROR, etc.)

## Troubleshooting

### Common Issues

1. **Query ID shows "N/A"**
   - Cause: Context not set properly
   - Solution: Ensure context managers are used correctly

2. **Duplicate Query IDs**
   - Cause: Manual ID generation without proper locking
   - Solution: Always use `query_id_manager` methods

3. **Context not propagating**
   - Cause: Context variables not set in current thread
   - Solution: Use context managers within the same thread

### Debugging Tips

1. **Filter logs by Query ID**: `grep "Query ID:2" logfile.log`
2. **Check context state**: Use `get_query_id()` and `get_main_query_id()`
3. **Monitor daily reset**: Check if counters reset at midnight

## Future Enhancements

Potential improvements for the logging system:

1. **Log Rotation**: Automatic log file rotation based on size
2. **Structured Logging**: JSON format for better parsing
3. **Metrics Collection**: Query performance metrics
4. **Remote Logging**: Send logs to external systems
5. **Log Levels per Module**: Different log levels for different modules
