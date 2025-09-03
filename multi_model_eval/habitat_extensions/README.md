# RxR Dataset for Habitat-Lab v0.2.4

A comprehensive implementation of the RxR (Room-across-Room) multilingual VLN dataset for Habitat-Lab v0.2.4, featuring flexible configuration management and seamless integration.

## Features

- 🌍 **Multilingual Support**: English (US/IN), Hindi, Telugu
- 👥 **Multi-Role Annotations**: Guide and follower perspectives
- 🔧 **Flexible Configuration**: File-based, environment variables, and defaults
- 📊 **Distributed Training**: Built-in chunking support
- 🎯 **Smart Filtering**: Scene, language, and episode-level filtering
- ✅ **Non-Intrusive**: Zero modifications to core Habitat-Lab

## Quick Start

### 1. Import the Extension

```python
import sys
sys.path.insert(0, '/path/to/multi_model_eval')

# Import RxR dataset (auto-registers with Habitat)
from habitat_extensions import RxRVLNDatasetV1
```

### 2. Basic Usage

```python
from omegaconf import DictConfig

# Create configuration
config = DictConfig({
    "type": "RxRVLN-v1",
    "split": "val_seen",
    "scenes_dir": "/path/to/scene_datasets/",
    "data_path": "/path/to/rxr/{split}_{role}.json.gz",
    "roles": ["guide", "follower"],
    "languages": ["en-US", "hi-IN"],
    "content_scenes": ["*"],
    "episodes_allowed": ["*"]
})

# Load dataset
dataset = RxRVLNDatasetV1(config)
print(f"Loaded {len(dataset.episodes)} episodes")
```

### 3. VLN Evaluation

```bash
# Single GPU
python multi_model_eval/vln_eval.py \
    --habitat_config_path config/vln_rxr.yaml \
    --eval_split val_unseen

# Multi-GPU
bash multi_model_eval/vln_eval_multi_gpu.sh
```

## Configuration Methods

### Method 1: Enhanced Configuration File

**File: `multi_model_eval/configs/vln_rxr_enhanced.yaml`**

```yaml
# Standard Habitat configuration
habitat:
  dataset:
    type: RxRVLN-v1
    split: val_seen
    scenes_dir: /path/to/scene_datasets/
    data_path: /path/to/rxr/{split}_{role}.json.gz

# RxR-specific parameters (avoids Habitat config conflicts)
rxr:
  roles: ["guide", "follower"]    # or ["*"] for all
  languages: ["*"]                # all languages
  content_scenes: ["*"]           # all scenes
  episodes_allowed: ["*"]         # all episodes
```

### Method 2: Environment Variables

```bash
# Configure via environment
export RXR_ROLES="guide,follower"
export RXR_LANGUAGES="en-US,hi-IN,te-IN"
export RXR_CONTENT_SCENES="*"

# Run evaluation
python multi_model_eval/vln_eval.py --habitat_config_path config/vln_rxr.yaml
```

### Method 3: Mixed Configuration

```bash
# Config file sets base parameters, env vars override specific ones
export RXR_LANGUAGES="hi-IN"  # Override only languages
python multi_model_eval/vln_eval.py --habitat_config_path config/vln_rxr.yaml
```

## Configuration Parameters

| Parameter | Type | Default | Description | Example |
|-----------|------|---------|-------------|---------|
| `roles` | List[str] | `["guide"]` | Annotation roles to load | `["guide", "follower"]`, `["*"]` |
| `languages` | List[str] | `["en-US"]` | Languages to load | `["en-US", "hi-IN"]`, `["*"]` |
| `content_scenes` | List[str] | `["*"]` | Scenes to load | `["scene1", "scene2"]`, `["*"]` |
| `episodes_allowed` | List[str] | `["*"]` | Episodes to load | `["1", "2", "3"]`, `["*"]` |
| `num_chunks` | int | - | Number of data chunks | `4` |
| `chunk_idx` | int | - | Current chunk index | `0` |

### Supported Values

**Roles**: `guide`, `follower`, `*` (all)
**Languages**: `en-US`, `en-IN`, `hi-IN`, `te-IN`, `*` (all)

## Extended Instruction Data

RxR episodes include rich metadata through `ExtendedInstructionData`:

```python
@dataclass
class ExtendedInstructionData:
    instruction_text: str        # Navigation instruction
    instruction_id: str          # Unique instruction ID
    language: str               # Language code (en-US, hi-IN, etc.)
    annotator_id: str          # Annotator identifier
    edit_distance: float       # Edit distance metric
    timed_instruction: List    # Temporal instruction data
    instruction_tokens: List   # Tokenized instruction
    split: str                 # Dataset split
```

## Usage Examples

### Loading Specific Languages

```python
# Load only Hindi instructions
config.languages = ["hi-IN"]
dataset = RxRVLNDatasetV1(config)

for episode in dataset.episodes:
    instruction = episode.instruction.instruction_text
    language = episode.instruction.language
    print(f"[{language}] {instruction}")
```

### Distributed Training

```python
# Load chunk 0 of 4 total chunks
config.num_chunks = 4
config.chunk_idx = 0
dataset = RxRVLNDatasetV1(config)
```

### Environment Variable Configuration

```bash
# Development: Quick testing with specific settings
export RXR_ROLES="guide"
export RXR_LANGUAGES="en-US"
python multi_model_eval/vln_eval.py --habitat_config_path config/vln_rxr.yaml

# Production: Use all data
export RXR_ROLES="*"
export RXR_LANGUAGES="*"
bash multi_model_eval/vln_eval_multi_gpu.sh
```

## Configuration Priority

The system uses a three-tier priority system:

```
Configuration File > Environment Variables > Default Values
```

This allows flexible deployment scenarios:
- **Development**: Override with environment variables for quick testing
- **Production**: Use configuration files for consistency
- **CI/CD**: Mix both approaches as needed

## Architecture

### Core Components

1. **RxRConfigExtension**: Configuration management system
   - Multi-source parameter merging
   - Automatic validation
   - Habitat integration

2. **RxRVLNDatasetV1**: Enhanced dataset implementation
   - Multi-role and multilingual support
   - Flexible filtering
   - Distributed training ready

3. **Smart Config Wrapper**: Intelligent configuration loading
   - Auto-detection of RxR configs
   - Seamless Habitat integration
   - Backward compatibility

### Integration Flow

```
Load Base Config → Detect RxR → Extract Parameters → 
Merge Environment → Validate → Inject to Habitat → Create Dataset
```

## Troubleshooting

### Import Errors
- Ensure Habitat-Lab v0.2.4 is installed
- Check Python path includes `multi_model_eval`
- Verify all dependencies are available

### Configuration Issues
```bash
# Debug configuration loading
python -c "
from habitat_extensions.config_utils import RxRConfigExtension
params = RxRConfigExtension.get_merged_rxr_config('config/vln_rxr.yaml')
print('Parameters:', params)
"
```

### Memory Issues
- Use `num_chunks` for large datasets
- Filter by specific scenes or languages
- Consider distributed evaluation

### Data Path Issues
- Verify data files exist at specified paths
- Check file permissions
- Ensure scene directories are accessible

## Best Practices

### Development
```bash
# Use environment variables for rapid iteration
export RXR_ROLES="guide"
export RXR_LANGUAGES="en-US"
python multi_model_eval/vln_eval.py --habitat_config_path config/vln_rxr.yaml
```

### Production
```bash
# Use configuration files for consistency
python multi_model_eval/vln_eval.py \
    --habitat_config_path multi_model_eval/configs/vln_rxr_enhanced.yaml
```

### Performance Optimization
- Use specific language/role filters to reduce memory usage
- Leverage distributed training with chunking
- Cache preprocessed data when possible

## File Structure

```
multi_model_eval/
├── habitat_extensions/
│   ├── __init__.py              # Extension registry
│   ├── rxr_dataset.py           # RxR dataset implementation
│   ├── config_utils.py          # Configuration management
│   └── README.md               # This documentation
├── configs/
│   ├── vln_rxr_enhanced.yaml   # Enhanced RxR configuration
│   └── ...
├── vln_eval.py                 # VLN evaluation script
└── vln_eval_multi_gpu.sh       # Multi-GPU evaluation script
```

## Compatibility

- **Habitat-Lab**: v0.2.4+
- **Python**: 3.7+
- **Dependencies**: `omegaconf`, `attrs`, `numpy`

## License

This extension follows the same MIT license as Habitat-Lab.

---

For additional examples and advanced usage, refer to the test files in `multi_model_eval/tests/` and example scripts in `multi_model_eval/examples/`.
