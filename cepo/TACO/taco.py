import json
import re

from datasets import load_dataset

from bespokelabs import curator


class TACOCodeExecutor(curator.CodeExecutor):
    """TACO Code Executor."""

    def code(self, row):
        """Extract code string from a dataset row, handling None and missing values safely."""
        execution = row.get("execution", "")  # Use .get() to avoid KeyError, default to ""

        try:
            if isinstance(execution, str) and "```python" in execution:
                match = re.search(r"```python\n(.*?)\n```", execution, re.DOTALL)
                return match.group(1) if match else execution  # Fallback to execution if no match found
            return execution
        except Exception as e:
            print(f"Error extracting code: {e}, row: {row}")  # Debugging log
            return ""

    def code_input(self, row):
        """Extract single input from a dataset row."""
        inputs_outputs = row["answer"]
        try:
            inputs_outputs = json.loads(inputs_outputs)
            inputs = inputs_outputs["inputs"][0]

        except Exception as e:
            print("Error parsing input output", e)
            inputs = ""

        if isinstance(inputs, list):
            inputs = "\n".join([str(i) for i in inputs])
        return inputs

    def code_output(self, row, execution_output):
        """Parse execution results."""
        inputs_outputs = row["answer"]
        try:
            inputs_outputs = json.loads(inputs_outputs)
            output = inputs_outputs["outputs"][0]
        except Exception as e:
            print("Error parsing input output", e)
            row["correct"] = "error"
            return row

        # Compare the output with execution stdout, stripping whitespace to handle formatting differences
        if isinstance(output, str) and isinstance(execution_output.stdout, str):
            row["correct"] = output.strip() == execution_output.stdout.strip()
        else:
            row["correct"] = output == execution_output.stdout
        return row


if __name__ == "__main__":
    executor = APPSCodeExecutor(backend="ray")
    dataset = load_dataset("bespokelabs/sky-t1-taco-test-rejection-sampled-shreyas")
    execution_output = executor(dataset["train"])

    print("================")
    print(execution_output)

    print(execution_output["correct"])
    print("================")