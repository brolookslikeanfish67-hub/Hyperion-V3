"""
Hyperion-V3: Autonomous Agent Tool Executor.
"""
import json
import re
from generate import HyperionInferenceEngine
from tools import HyperionTools

class HyperionAgentExecutor:
    def __init__(self, engine: HyperionInferenceEngine):
        self.engine = engine
        self.tools = HyperionTools()

    def run_tool_loop(self, user_prompt: str, max_steps: int = 5) -> str:
        """
        Orchestrates an interactive, autonomous ReAct tool execution chain loop.
        Ensures clear boundaries between internal reasoning steps and system feedback loops.
        """
        system_instructions = (
            "System Instructions:\n"
            "You are an elite autonomous agent. You must solve the user task using step-by-step reasoning.\n"
            "You have access to the following tools:\n"
            "1. Read File:   [READ:path]\n"
            "2. Write File:  [WRITE:path|content]\n"
            "3. Run Python:  [EXEC:code]\n"
            "4. Terminal:    [CMD:command]\n\n"
            "To invoke a tool, you MUST structure your response precisely as:\n"
            "An optional <|thinking|> reasoning trace block, followed by the exact tool call token tag.\n"
            "Do not mix multiple tools in one step. Wait for the tool output before proceeding."
        )

        # Initialize linear dialog stream state
        current_context = (
            f"{system_instructions}\n\n"
            f"User Task: {user_prompt}\n\n"
            f"Assistant: <|thinking|>\n"
        )

        for step in range(max_steps):
            print(f" [Agent] Tool Step Execution: #{step + 1}/{max_steps}")
            
            # Generate next reasoning stream token sequence with low temperature for deterministic routing
            response = self.engine.generate(
                current_context, 
                max_new_tokens=512, 
                temperature=0.1
            ).strip()
            
            print(f" [Model Thought Trace]:\n{response}\n")

            # Check for structural tool execution hooks using non-greedy multiline tracking
            tool_call = re.search(r"\[(READ|WRITE|EXEC|CMD)\s*:\s*(.*?)\]", response, re.DOTALL)
            
            if not tool_call:
                print(" [Agent] Loop terminated: No valid tool tags found. Returning final answer.")
                # Ensure the conversation context captures the final generated text
                current_context += f"{response}\n"
                return response

            tool_type = tool_call.group(1)
            tool_args = tool_call.group(2)
            tool_output = ""

            print(f"🛠️ [Calling Tool]: {tool_type}")
            
            try:
                if tool_type == "READ":
                    tool_output = self.tools.read_local_file(tool_args.strip())
                elif tool_type == "WRITE":
                    if "|" in tool_args:
                        path, content = tool_args.split("|", 1)
                        tool_output = self.tools.write_local_file(path.strip(), content)
                    else:
                        tool_output = "Error: Invalid WRITE format. Missing character symbol partition delimiter (|)."
                elif tool_type == "EXEC":
                    tool_output = self.tools.execute_python_code(tool_args)
                elif tool_type == "CMD":
                    tool_output = self.tools.run_terminal_command(tool_args.strip())
            except Exception as e:
                tool_output = f"Runtime Crash Exception: {str(e)}"

            print(f" [Tool Output Logs]:\n{tool_output}\n")

            # Update dialog flow state safely without re-appending base system prompts.
            # Explicitly close the reasoning trace block and append environmental feedback.
            current_context += (
                f"{response}\n"
                f"</|thinking|>\n"
                f"Tool Observation Logs:\n{tool_output.strip()}\n"
                f"Next Action: <|thinking|>\n"
            )

        print(" [Agent] Execution warning: Maximum step limit reached.")
        return "Error: Maximum execution step bounds exceeded."

if __name__ == "__main__":
    from generate import CHECKPOINT_PATH, TOKENIZER_JSON

    # Initialize your base model layer graph mapping configurations
    base_engine = HyperionInferenceEngine(
        checkpoint_path=CHECKPOINT_PATH, 
        tokenizer_path=TOKENIZER_JSON
    )
    
    executor = HyperionAgentExecutor(base_engine)
    
    # Run the interactive verification loop
    task = "Write a python script to verify matrix multiplication and test it."
    final_output = executor.run_tool_loop(task)
