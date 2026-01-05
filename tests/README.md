# Fun-ASR 测试脚本

本目录包含性能测试和优化相关的脚本。

## 测试脚本

| 脚本 | 说明 |
|------|------|
| `performance_test.py` | 基础性能测试（需要 ffmpeg） |
| `quick_performance_test.py` | 快速性能测试（无外部依赖） |
| `full_performance_test.py` | 完整批量测试（1-20） |
| `test_remote_gpu.py` | 远程 GPU 服务器性能测试 |
| `test_remote_rtf.py` | 远程服务器 RTF 对比测试 |
| `optimize_l20.py` | L20 GPU 优化测试（批量1-50，并发测试） |

## 使用方法

```bash
# 确保服务已启动
curl http://localhost:8000/health

# 运行快速测试
python quick_performance_test.py

# 运行完整测试
python full_performance_test.py

# 测试远程服务器
python test_remote_gpu.py

# L20 优化测试
python optimize_l20.py
```

## 测试音频

`aishell_test/` 目录包含测试音频样本。
