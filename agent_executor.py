"""
Hyperion-V3: Autonomous Agent Tool Executor.
"""
import json
import re
from generate import HyperionInferenceEngine
from tools import HyperionTools

class HyperionAgentExecutor:
    def __init__(
        self, engine: HyperionInferenceEngine
    ):
        self.engine = engine
        self.tools = HyperionTools()

    def run_tool_loop(
        self, user_prompt: str, max_steps: int = 5
    ) -> str:
        """
        Orchestrates an interactive, autonomous
        ReAct tool execution chain loop.
        """
        system_instructions = (
            "You have access to tools:\n"
            "1. Read File: [READ:path]\n"
            "2. Write File: [WRITE:path|content]\n"
            "3. Run Python: [EXEC:code]\n"
            "4. Terminal: [CMD:command]\n"
            "To use a tool, output the exact tag format."
        )
        
        current_context = (
            f"{system_instructions}\n"
            f"User Task: {user_prompt}\n"
            f"Assistant: <|thinking|>\n"
        )
        
        for step in range(max_steps):
            print(f"[Agent] Tool Step Matrix: #{step + 1}")
            
            # Generate next reasoning stream token sequence
            response = self.engine.generate(
                current_context,
                max_new_tokens=256,
                temperature=0.1
            )
            
            print(f"[Model Thought]:\n{response}\n")
            
            # Check for structural tool execution hooks
            tool_call = re.search(
                r"\[(READ|WRITE|EXEC|CMD):(.*?)\]", 
                response, 
                re.DOTALL
            )
            
            if not tool_call:
                # No tool called, return the final response string
                return response
                
            tool_type = tool_call.group(1)
            tool_args = tool_call.group(2)
            tool_output = ""
            
            print(f"[ Calling Tool]: {tool_type}")
            
            if tool_type == "READ":
                tool_output = self.tools.read_local_file(
                    tool_args.strip()
                )
            elif tool_type == "WRITE":
                if "|" in tool_args:
                    path, content = tool_args.split("|", 1)
                    tool_output = self.tools.write_local_file(
                        path.strip(), content
                    )
                else:
                    tool_output = "Error: Invalid WRITE format."
            elif tool_type == "EXEC":
                tool_output = self.tools.execute_python_code(
                    tool_args
                )
            elif tool_type == "CMD":
                tool_output = self.tools.run_terminal_command(
                    tool_args.strip()
                )
                
            print(f"[ Tool Output Logs]:\n{tool_output}\n")
            
            # Append live environment feedback to context buffer
            current_context += (
                f"{response}\n"
                f"Tool Observation Logs:\n{tool_output}\n"
                f"Next Action: <|thinking|>\n"
            )
            
        return "Error: Maximum execution step bounds exceeded."

if __name__ == "__main__":
    from generate import CHECKPOINT_PATH, TOKENIZER_JSON
    
    # Initialize your base model layer graph mapping configurations
    base_engine = HyperionInferenceEngine(
        checkpoint_path=CHECKPOINT_PATH,
        tokenizer_path=TOKENIZER_JSON
    )
    
    executor = HyperionAgentExecutor(base_engine)
    
    # Example task requiring multiple system tools
    task = "Write a python script to verify matrix multiplication."
    executor.run_tool_loop(task)
