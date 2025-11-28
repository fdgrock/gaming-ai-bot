# Testing Infrastructure for Gaming AI Bot

This directory contains comprehensive testing infrastructure for the Gaming AI Bot system after **Phase 2 Service Extraction**. All test files are properly organized here to maintain clean project structure.

## 📁 Directory Structure

```
tests/
├── conftest.py                    # Central pytest configuration and fixtures
├── fixtures/
│   └── sample_data.py            # Test data and sample fixtures
├── unit/                         # Unit tests for individual components
│   ├── test_services/            # Service layer tests
│   │   ├── test_data_service.py
│   │   ├── test_prediction_service.py
│   │   └── test_cache_service.py
│   └── test_ai_engines/          # AI engine tests
│       └── test_ensemble_engine.py
├── integration/                  # Integration and workflow tests
│   └── test_prediction_workflow.py
├── test_all_services.py         # Comprehensive Phase 2 service tests
├── test_runner.py               # Advanced test runner with reporting
├── test_imports.py              # Service import validation
├── test_connectivity.py         # System connectivity tests
├── test_streamlit.py           # UI layer tests
├── validate_phase2.py          # Phase 2 completion validation
└── README.md                   # This file
```

## 🎯 Phase 2 Testing Coverage

### Service Layer Testing (Phase 2 Complete)
- **DataService**: Tests for 25+ extracted data management functions
- **ModelService**: Tests for 15+ ML model management functions  
- **PredictionService**: Tests for 20+ prediction generation functions
- **AnalyticsService**: Tests for 15+ analytics and trend analysis functions
- **TrainingService**: Tests for 19+ ML training workflow functions
- **ServiceRegistry**: Tests for dependency injection and lifecycle management

### Test Organization Principles
✅ **All tests in tests/ folder** - No test files scattered in project root  
✅ **Logical categorization** - Unit, integration, validation organized separately  
✅ **Service-focused testing** - Each extracted service has comprehensive test coverage  
✅ **Automated test running** - Advanced test runner with detailed reporting  
✅ **Phase validation** - Specific validation for Phase 2 completion  

## 🧪 Testing Framework Features

### Advanced Test Runner (test_runner.py)
The test runner provides comprehensive testing capabilities with detailed reporting:

```bash
# Run all tests with detailed report
python tests/test_runner.py

# Run quick smoke tests for rapid validation
python tests/test_runner.py smoke

# Run tests for specific service
python tests/test_runner.py data
python tests/test_runner.py model
python tests/test_runner.py prediction
python tests/test_runner.py analytics
python tests/test_runner.py training

# Save detailed results to JSON file
python tests/test_runner.py all --save
```

### Test Categories

#### Service Layer Tests (test_all_services.py)
- **Comprehensive Coverage**: 94+ functions across 5 services tested
- **Unit Testing**: Each service method tested in isolation
- **Integration Testing**: Service interactions and workflows validated
- **Mock Support**: Proper mocking of dependencies and external systems

#### Validation Tests
- **test_imports.py**: Validates all service imports work correctly
- **validate_phase2.py**: Comprehensive Phase 2 completion validation
- **test_connectivity.py**: System connectivity and dependency validation

#### Legacy Tests (Pre-Phase 2)
- **unit/test_services/**: Original service tests (now complemented by Phase 2 tests)
- **integration/**: System-wide integration tests
- **test_streamlit.py**: UI layer testing

### Central Configuration (conftest.py)
- **50+ Fixtures**: Comprehensive mock services, sample data, and test utilities
- **Mock Services**: Complete mocking for all service layers
- **Sample Data**: Realistic lottery data, predictions, statistics, and configurations
- **Database Fixtures**: In-memory SQLite databases for testing
- **AI Engine Mocks**: Mock implementations of all AI engines
- **Performance Testing**: Timer fixtures and async testing support
- **Parametrized Testing**: Test data for multiple games and strategies

### Test Categories

#### Unit Tests
- **Service Tests**: Data service, prediction service, cache service
- **AI Engine Tests**: Ensemble engine, neural network, pattern recognition
- **Component Tests**: Chart components, table components, UI widgets
- **Page Tests**: Main page, analysis page, settings page
- **Configuration Tests**: App config, game config validation

#### Integration Tests
- **Prediction Workflow**: End-to-end prediction generation
- **Data Flow**: Service interaction and data consistency
- **Cache Integration**: Caching behavior across components
- **Error Handling**: Error propagation and recovery mechanisms
- **Performance**: Load testing and stress testing

## 🔧 Development Tools

### Debug Utils (`tools/dev_tools.py`)
```python
from tools.dev_tools import DebugUtils

debug = DebugUtils()

# Trace function calls
@debug.trace_function_calls
def my_function():
    pass

# Capture component state
debug.capture_state("service_name", {"key": "value"})

# Performance analysis
analysis = debug.analyze_performance(performance_data)
```

### Validation Tools (`tools/validation_utils.py`)
```python
from tools.validation_utils import DataValidator, TestDataGenerator

# Validate lottery data
validator = DataValidator()
result = validator.validate_lottery_numbers([1, 15, 25, 35, 45], "powerball")

# Generate test data
generator = TestDataGenerator()
historical_data = generator.generate_historical_data("powerball", days=365)
predictions = generator.generate_predictions("powerball", count=10)
```

### Performance Testing (`tools/performance_testing.py`)
```python
from tools.performance_testing import PerformanceTester, SystemBenchmark

# Performance testing
tester = PerformanceTester()
result = tester.benchmark_function(my_function, iterations=100)

# System benchmarking
benchmark = SystemBenchmark()
results = benchmark.full_system_benchmark(services)
```

## 🚀 Running Tests

### Prerequisites
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_services/test_data_service.py

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Advanced Testing Options
```bash
# Run tests in parallel
pytest -n auto

# Run tests with specific markers
pytest -m "not slow"

# Run failed tests only
pytest --lf

# Generate detailed test report
pytest --html=report.html --self-contained-html
```

## 📊 Test Data and Fixtures

### Sample Data Types
- **Historical Data**: 365 days of realistic lottery draws
- **Predictions**: Various strategies and engines
- **Statistics**: Frequency analysis, patterns, trends
- **User Preferences**: Different user configurations
- **Performance Metrics**: System performance data
- **Game Configurations**: Multiple lottery games

### Fixture Categories
- **Mock Services**: `mock_data_service`, `mock_prediction_service`, `mock_cache_service`
- **Sample Data**: `sample_historical_data`, `sample_predictions`, `sample_statistics`
- **Database**: `in_memory_db`, `populated_test_db`
- **AI Engines**: `mock_ensemble_engine`, `mock_neural_engine`
- **Configuration**: `test_config`, `sample_game_config`
- **Performance**: `performance_timer`, `async_timer`

## 🎯 Test Coverage Areas

### Functional Testing
- ✅ Data retrieval and storage
- ✅ Prediction generation algorithms
- ✅ Cache operations and TTL
- ✅ User preference management
- ✅ Statistical analysis
- ✅ Error handling and recovery

### Performance Testing
- ✅ Response time benchmarking
- ✅ Memory usage profiling
- ✅ Concurrent access testing
- ✅ Load testing with varying users
- ✅ Stress testing under high load
- ✅ Cache performance optimization

### Integration Testing
- ✅ Service-to-service communication
- ✅ End-to-end workflows
- ✅ Data consistency across components
- ✅ Real-time prediction generation
- ✅ Batch processing operations
- ✅ Error propagation and handling

## 🔍 Test Validation

### Data Validation
- Number range validation (1-69 for Powerball)
- Duplicate number detection
- Date format validation
- Confidence score validation (0-1)
- Strategy and engine validation

### System Health Validation
- Service responsiveness
- Memory usage monitoring
- Error rate tracking
- Performance degradation detection
- Database integrity checks

## 📈 Performance Benchmarks

### Target Metrics
- **Prediction Generation**: < 2 seconds
- **Data Retrieval**: < 100ms
- **Cache Operations**: < 10ms
- **Memory Usage**: < 512MB
- **Error Rate**: < 1%
- **Cache Hit Rate**: > 80%

### Benchmark Tests
```python
# Example benchmark usage
benchmark = SystemBenchmark()
results = benchmark.full_system_benchmark({
    "data_service": data_service,
    "prediction_service": prediction_service,
    "cache_service": cache_service
})
```

## 🛠️ Development Workflow

### 1. Test-Driven Development
```bash
# Write failing test
pytest tests/unit/test_new_feature.py -v

# Implement feature
# Run test to ensure it passes
pytest tests/unit/test_new_feature.py -v
```

### 2. Integration Testing
```bash
# Test complete workflows
pytest tests/integration/ -v

# Test with different configurations
pytest tests/integration/ --config-file=test_config.json
```

### 3. Performance Validation
```bash
# Run performance tests
python tools/performance_testing.py

# Generate performance report
pytest tests/ --benchmark-only
```

## 📋 Testing Checklist

### Before Deployment
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Data validation tests pass
- [ ] Error handling tests pass
- [ ] Memory leak tests pass
- [ ] Security validation passes

### Continuous Integration
- [ ] Automated test execution
- [ ] Code coverage reporting
- [ ] Performance regression detection
- [ ] Test result notifications
- [ ] Failure investigation logs

## 🐛 Debugging and Troubleshooting

### Common Issues
1. **Test Fixtures Not Loading**: Check `conftest.py` imports
2. **Mock Services Failing**: Verify mock configurations
3. **Performance Tests Slow**: Reduce iteration counts for development
4. **Database Tests Failing**: Ensure in-memory database setup

### Debug Commands
```bash
# Debug specific test
pytest tests/unit/test_services/test_data_service.py::TestDataService::test_get_historical_data_success -v -s

# Run with debugger
pytest --pdb tests/unit/test_services/test_data_service.py

# Verbose output with logging
pytest -v -s --log-cli-level=DEBUG
```

## 📚 Additional Resources

### Documentation
- [Pytest Documentation](https://docs.pytest.org/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)

### Best Practices
- Write descriptive test names
- Use meaningful assertions
- Keep tests independent
- Mock external dependencies
- Test edge cases and error conditions
- Maintain high test coverage (>90%)

---

**Note**: This testing infrastructure provides comprehensive coverage for the lottery prediction system. All tests are designed to be fast, reliable, and maintainable. The framework supports both development testing and continuous integration scenarios.