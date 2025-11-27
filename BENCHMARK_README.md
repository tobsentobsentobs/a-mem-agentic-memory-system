# Ollama Model Benchmark Tool

Modern TUI (Text User Interface) for speed testing of Ollama models.

## 🚀 Features

- ✅ **Modern TUI** with Textual Framework
- ✅ **Live Metrics**: Tokens/sec, Latency, First Token Time
- ✅ **Multi-Model Testing**: Test different models sequentially
- ✅ **Results Export**: Save results as JSON
- ✅ **Real-time Progress**: Live progress bar during benchmark
- ✅ **Interactive Logs**: Detailed logs for each test

## 📋 Installation

```bash
# Install Dependencies
pip install textual requests

# Or use requirements.txt
pip install -r requirements.txt
```

## 🎯 Usage

```bash
# Start the benchmark tool
python ollama_benchmark.py
```

### TUI Navigation

- **Select Model**: Dropdown menu at the top
- **Adjust Prompt**: Text input for test prompt
- **Start Benchmark**: 
  - Click button "🚀 Run Benchmark"
  - Or press key `r`
- **Clear Results**: 
  - Click button "🗑️ Clear Results"
  - Or press key `c`
- **Save Results**: 
  - Click button "💾 Save Results"
  - Or press key `s`
- **Exit**: Press key `q`

## 📊 Metrics

The tool measures the following performance metrics:

- **Tokens/sec**: Generation speed
- **Total Time**: Total response time
- **Tokens**: Number of generated tokens
- **First Token Time**: Time to first token (TTFT)
- **Avg Token Time**: Average time per token

## 💾 Export

Results are saved as JSON:

```json
[
  {
    "model": "qwen3:4b",
    "prompt": "Write a short story...",
    "total_time": 12.34,
    "tokens_generated": 87,
    "tokens_per_second": 7.05,
    "first_token_time": 0.234,
    "avg_token_time": 0.142,
    "timestamp": "2025-11-27T04:00:00"
  }
]
```

## 🎨 Screenshots

The TUI shows:
- Model selector
- Prompt editor
- Live progress bar
- Results table with all metrics
- Detailed logs

## 🔧 Customization

### Custom Prompts

You can change the default prompt in the code:

```python
current_prompt = reactive("Your custom prompt here...")
```

### Max Tokens

Default: 100 tokens. Change in `run_benchmark()`:

```python
result = await loop.run_in_executor(
    None,
    benchmark.benchmark_model,
    self.current_model,
    self.current_prompt,
    200  # adjust max_tokens
)
```

## 🐛 Troubleshooting

**"No models found"**
- Make sure Ollama is running: `ollama serve`
- Check if models are installed: `ollama list`

**"Connection refused"**
- Check Ollama URL (default: `http://localhost:11434`)
- Change `OLLAMA_BASE_URL` in code if necessary

**Benchmark hangs**
- Check Ollama logs
- Make sure enough RAM/VRAM is available

## 📝 Example Output

```
Model          | Tokens/sec | Total Time | Tokens | First Token | Avg Token
qwen3:4b       | 7.05       | 12.34      | 87     | 0.234       | 142.00
llama3.2:3b    | 12.45      | 8.03       | 100    | 0.189       | 80.30
mistral:7b     | 5.23       | 19.12      | 100    | 0.456       | 191.20
```

## 🎯 Best Practices

1. **Warm-up**: First request may be slower (Model Loading)
2. **Consistency**: Use the same prompt for fair comparisons
3. **Multiple Runs**: Run multiple benchmarks for average values
4. **System Load**: Close other GPU-intensive apps during tests

## 📚 Technical Details

- **Framework**: Textual (modern Python TUI library)
- **API**: Ollama REST API (`/api/generate`)
- **Streaming**: Uses streaming for precise token measurement
- **Async**: Asynchronous execution for responsive UI

## 🔗 Links

- [Textual Documentation](https://textual.textualize.io/)
- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Models](https://ollama.com/library)
