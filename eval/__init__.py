"""eval 包：评测体系（case 定义、运行器、指标计算、消融实验）。

将 eval/ 声明为正式 Python 包，使得 `from eval.eval_core import ...` 和
`from eval import run_eval` 在 IDE 和测试中均可正确解析，无需 sys.path hack。
"""
