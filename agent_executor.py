"""
Hyperion-V3: Autonomous Agent Tool Executor.
Includes native multi-step socket-level crawling boundaries.
"""
import json
import re
from generate import HyperionInferenceEngine
from tools import HyperionTools

class HyperionAgentExecutor:
    def __init__(self, engine: HyperionInferenceEngine):
        self.engine = engine
        self.tools = HyperionTools()

    def run_tool_loop(self, user_prompt: str, max_steps: int = 10) -> str:
        """
        Orchestrates an interactive, autonomous ReAct tool execution chain loop.
        Calibrated to force the model to continuously browse and crawl deep web paths.
        """
        system_instructions = (
            "System Instructions:\n"
            "You are an elite autonomous research and coding agent.\n"
            "You must solve the user task using comprehensive, step-by-step reasoning.\n"
            "You have access to the following native tools:\n"
            "1. Read Local File:   [READ:path]\n"
            "2. Write Local File:  [WRITE:path|content]\n"
            "3. Run Python:        [EXEC:code]\n"
            "4. Terminal Command:  [CMD:command]\n"
            "5. Socket Web Browse: [BROWSE:url]\n"
            "6. Extract Hyperlinks:[EXTRACT_LINKS:url]\n\n"
            "CRAWLING DIRECTIVES:\n"
            "- Never guess or rely on stale pre-trained weights for live information.\n"
            "- You are expected to use [BROWSE:url] and [EXTRACT_LINKS:url] frequently.\n"
            "- Always parse a directory page, extract its outgoing hyperlinks, and browse "
            "nested sub-paths sequentially to collect deep validation context.\n\n"
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
            print(f"🤖 [Agent] Tool Step Execution: #{step + 1}/{max_steps}")
            
            # Generate next reasoning stream token sequence with low temperature for deterministic routing
            response = self.engine.generate(
                current_context, 
                max_new_tokens=512, 
                temperature=0.1
            ).strip()
            
            print(f"🧠 [Model Thought Trace]:\n{response}\n")

            # Check for structural tool execution hooks using non-greedy multiline tracking
            # Supports READ, WRITE, EXEC, CMD, BROWSE, and EXTRACT_LINKS
            tool_call = re.search(r"\[(READ|WRITE|EXEC|CMD|BROWSE|EXTRACT_LINKS)\s*:\s*(.*?)\]", response, re.DOTALL)
            
            if not tool_call:
                print("🏁 [Agent] Loop terminated: No valid tool tags found. Returning final answer.")
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
                        tool_output = "Error: Invalid WRITE format. Missing partition delimiter (|)."
                elif tool_type == "EXEC":
                    tool_output = self.tools.execute_python_code(tool_args)
                elif tool_type == "CMD":
                    tool_output = self.tools.run_terminal_command(tool_args.strip())
                elif tool_type == "BROWSE":
                    tool_output = self.tools.raw_http_get(tool_args.strip())
                elif tool_type == "EXTRACT_LINKS":
                    tool_output = self.tools.extract_links_from_content(tool_args.strip(), current_context)
            except Exception as e:
                tool_output = f"Runtime Crash Exception: {str(e)}"

            print(f"📝 [Tool Output Logs]:\n{tool_output}\n")

            # Update dialog flow state safely without re-appending base system prompts.
            # Explicitly close the reasoning trace block and append environmental feedback.
            current_context += (
                f"{response}\n"
                f"</|thinking|>\n"
                f"Tool Observation Logs:\n{tool_output.strip()}\n"
                f"Next Action: <|thinking|>\n"
            )

        print("🚨 [Agent] Execution warning: Maximum step limit reached.")
        return "Error: Maximum execution step bounds exceeded."


if __name__ == "__main__":
    from generate import CHECKPOINT_PATH, TOKENIZER_JSON

    # Initialize your base model layer graph mapping configurations
    base_engine = HyperionInferenceEngine(
        checkpoint_path=CHECKPOINT_PATH, 
        tokenizer_path=TOKENIZER_JSON
    )
    
    executor = HyperionAgentExecutor(base_engine)
    
    # Run the interactive multi-step crawler test verification loop
    task = "Browse https://ycombinator.com, find the top link, extract its paths, and scrape its body."
    final_output = executor.run_tool_loop(task, max_steps=10)
