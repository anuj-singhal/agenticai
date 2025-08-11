import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import AsyncExitStack
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import gradio as gr
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentMemory:
    """Memory structure for the agent"""
    task_history: List[Dict[str, Any]]
    learned_patterns: Dict[str, Any]
    successful_tool_combinations: List[List[str]]
    failed_attempts: List[Dict[str, Any]]
    resource_cache: Dict[str, Any]
    
    def add_task(self, task: str, result: Any, tools_used: List[str], success: bool):
        """Add a completed task to memory"""
        self.task_history.append({
            'timestamp': datetime.now().isoformat(),
            'task': task,
            'result': result,
            'tools_used': tools_used,
            'success': success
        })
        
        if success and tools_used:
            self.successful_tool_combinations.append(tools_used)
        elif not success:
            self.failed_attempts.append({
                'task': task,
                'tools_used': tools_used,
                'timestamp': datetime.now().isoformat()
            })

class MCPAgent:
    """Autonomous MCP Client Agent with feedback loop and memory"""
    
    def __init__(self, openai_api_key: str, server_configs: List[Dict[str, Any]]):
        """
        Initialize the MCP Agent
        
        Args:
            openai_api_key: OpenAI API key
            server_configs: List of MCP server configurations
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.server_configs = server_configs
        self.sessions = {}
        self.memory = AgentMemory(
            task_history=[],
            learned_patterns={},
            successful_tool_combinations=[],
            failed_attempts=[],
            resource_cache={}
        )
        self.available_tools = {}
        self.available_tools_list = []
        self.available_prompts = {}
        self.available_resources = {}
        self.tool_name_to_server = {}  # Map tool names to their servers
        self.initialized = False
        
    async def initialize_servers(self):
        """Initialize connections to all MCP servers"""
        if self.initialized:
            return
            
        for config in self.server_configs:
            print(f"Initializing server: {config['name']}")
            try:
                server_params = StdioServerParameters(
                    command=config['command'],
                    args=config.get('args', []),
                    env=config.get('env', None)
                )
                print("server_params", server_params)

                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                self.stdio, self.write = stdio_transport
                async_session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                
                await async_session.initialize()
                
                # List available tools
                response = await async_session.list_tools()
                tools = response.tools
                print("\nConnected to server with tools:", [tool.name for tool in tools])

                self.sessions[config['name']] = async_session
                logger.info(f"Connected to MCP server: {config['name']}")
                
                # Discover capabilities
                await self._discover_server_capabilities(config['name'], async_session)
                
            except Exception as e:
                logger.error(f"Failed to connect to server {config['name']}: {e}")
        
        self.initialized = True
    
    async def _discover_server_capabilities(self, server_name: str, session: ClientSession):
        """Discover tools, prompts, and resources from an MCP server"""
        try:
            # Get available tools
            tools_result = await session.list_tools()
            if tools_result.tools:
                self.available_tools[server_name] = {
                    tool.name: {
                        'description': tool.description,
                        'input_schema': tool.inputSchema
                    } for tool in tools_result.tools
                }

            # Get available tools list for OpenAI format
            if tools_result.tools:
                for tool in tools_result.tools:
                    self.available_tools_list.append({ 
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                    # Map tool name to server for easy lookup
                    self.tool_name_to_server[tool.name] = server_name
                    
                logger.info(f"Discovered {len(tools_result.tools)} tools from {server_name}")
            
            # Get available prompts
            prompts_result = await session.list_prompts()
            if prompts_result.prompts:
                self.available_prompts[server_name] = {
                    prompt.name: {
                        'description': prompt.description,
                        'arguments': prompt.arguments
                    } for prompt in prompts_result.prompts
                }
                logger.info(f"Discovered {len(prompts_result.prompts)} prompts from {server_name}")
            
            # Get available resources
            resources_result = await session.list_resources()
            if resources_result.resources:
                self.available_resources[server_name] = {
                    resource.uri: {
                        'name': resource.name,
                        'description': resource.description,
                        'mimeType': resource.mimeType
                    } for resource in resources_result.resources
                }
                logger.info(f"Discovered {len(resources_result.resources)} resources from {server_name}")
                
        except Exception as e:
            logger.error(f"Error discovering capabilities for {server_name}: {e}")
    
    async def _plan_task_execution(self, task: str) -> Dict[str, Any]:
        """Use LLM to create an execution plan for the task"""
        capabilities_context = self._build_capabilities_context()
        memory_context = self._build_memory_context()
        
        planning_prompt = f"""
        You are an intelligent agent that needs to create an execution plan for the following task using available MCP tools.
        
        TASK: {task}
        
        AVAILABLE TOOLS AND CAPABILITIES:
        {capabilities_context}
        
        MEMORY FROM PAST TASKS:
        {memory_context}
        
        Create a step-by-step execution plan. For each step that requires a tool, specify:
        1. The exact tool name
        2. The arguments needed for that tool
        3. Why this tool is needed
        
        Provide a JSON response with:
        {{
            "plan": [
                {{
                    "step": 1,
                    "description": "Description of what this step does",
                    "tool_name": "exact_tool_name or null if no tool needed",
                    "arguments": {{"arg1": "value1", "arg2": "value2"}} or null,
                    "reasoning": "Why this step is needed"
                }}
            ],
            "confidence": 0.8,
            "overall_strategy": "Brief description of the overall approach"
        }}
        
        Return ONLY valid JSON.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.3,
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            return {
                "plan": [{"step": 1, "description": "fallback", "tool_name": None, "arguments": None, "reasoning": "Fallback due to parsing error"}],
                "confidence": 0.1,
                "overall_strategy": "Fallback strategy"
            }
    
    async def _execute_with_openai_tool_calling(self, task: str) -> Tuple[Any, bool, List[str]]:
        """Execute task using OpenAI's tool calling feature"""
        messages = [
            {
                "role": "system", 
                "content": f"""You are an AI assistant that can use various tools to complete tasks. 
                You have access to weather tools, database tools, and financial tools.
                
                Current task: {task}
                
                Use the available tools to complete this task step by step. 
                Make sure to call tools with the correct arguments based on their schemas."""
            },
            {"role": "user", "content": task}
        ]
        
        tools_used = []
        results = []
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.available_tools_list,
                tool_choice="auto",
                temperature=0.3
            )
            
            message = response.choices[0].message
            messages.append(message.dict())
            
            # Check if the model wants to call tools
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
                    
                    try:
                        # Find which server has this tool
                        server_name = self.tool_name_to_server.get(tool_name)
                        if not server_name:
                            raise ValueError(f"Tool {tool_name} not found in any server")
                        
                        # Execute the tool
                        result = await self._use_tool(server_name, tool_name, arguments, task)
                        results.append(result)
                        tools_used.append(f"{server_name}.{tool_name}")
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result)
                        })
                        
                    except Exception as e:
                        error_msg = f"Error calling tool {tool_name}: {str(e)}"
                        logger.error(error_msg)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg
                        })
            else:
                # No more tool calls, the task is complete
                final_response = message.content
                results.append(final_response)
                break
        
        return results, True, tools_used
    
    def _build_capabilities_context(self) -> str:
        """Build context string of available capabilities"""
        context = "AVAILABLE TOOLS:\n"
        for server, tools in self.available_tools.items():
            context += f"\nServer: {server}\n"
            for tool_name, tool_info in tools.items():
                context += f"  - {tool_name}: {tool_info['description']}\n"
                if 'input_schema' in tool_info and tool_info['input_schema']:
                    properties = tool_info['input_schema'].get('properties', {})
                    if properties:
                        context += f"    Parameters: {list(properties.keys())}\n"
        
        context += "\nAVAILABLE PROMPTS:\n"
        for server, prompts in self.available_prompts.items():
            for prompt_name, prompt_info in prompts.items():
                context += f"- {server}.{prompt_name}: {prompt_info['description']}\n"
        
        context += "\nAVAILABLE RESOURCES:\n"
        for server, resources in self.available_resources.items():
            for uri, resource_info in resources.items():
                context += f"- {server}: {resource_info['name']} - {resource_info['description']}\n"
        
        return context
    
    def _build_memory_context(self) -> str:
        """Build context from agent's memory"""
        context = f"SUCCESSFUL TOOL COMBINATIONS: {self.memory.successful_tool_combinations[-5:]}\n"
        context += f"RECENT TASKS: {[t['task'] for t in self.memory.task_history[-3:]]}\n"
        context += f"LEARNED PATTERNS: {list(self.memory.learned_patterns.keys())[-3:]}\n"
        return context
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> Tuple[Any, bool]:
        """Execute the planned steps"""
        results = []
        tools_used = []
        
        try:
            for step_info in plan['plan']:
                step_num = step_info['step']
                description = step_info['description']
                tool_name = step_info.get('tool_name')
                arguments = step_info.get('arguments', {})
                
                logger.info(f"Executing Step {step_num}: {description}")
                
                if tool_name:
                    # Find which server has this tool
                    server_name = self.tool_name_to_server.get(tool_name)
                    if not server_name:
                        # Try to find the tool in available tools
                        found = False
                        for server, tools in self.available_tools.items():
                            if tool_name in tools:
                                server_name = server
                                found = True
                                break
                        
                        if not found:
                            raise ValueError(f"Tool {tool_name} not found in any server")
                    
                    # Execute the tool
                    result = await self._use_tool(server_name, tool_name, arguments or {}, description)
                    results.append(result)
                    tools_used.append(f"{server_name}.{tool_name}")
                    
                    logger.info(f"Step {step_num} completed with result: {str(result)[:200]}")
                else:
                    # This is a planning or informational step
                    results.append(f"Completed: {description}")
                    logger.info(f"Step {step_num} completed: {description}")
            
            return results, True
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return str(e), False
    
    async def _use_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any], task: str) -> Any:
        """Use a specific tool from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        
        # Ensure arguments is a dictionary
        if not isinstance(arguments, dict):
            arguments = {}
        
        logger.info(f"Calling tool {tool_name} on server {server_name} with arguments: {arguments}")
        
        result = await session.call_tool(tool_name, arguments)
        return result.content if result.content else "Tool executed successfully"
    
    async def _access_resource(self, server_name: str, resource_uri: str) -> Any:
        """Access a specific resource from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.read_resource(resource_uri)
        return result.contents if result.contents else "Resource accessed successfully"
    
    async def _use_prompt(self, server_name: str, prompt_name: str, arguments: Dict[str, Any]) -> Any:
        """Use a specific prompt from an MCP server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server {server_name} not connected")
        
        session = self.sessions[server_name]
        result = await session.get_prompt(prompt_name, arguments)
        return result.messages if result.messages else "Prompt executed successfully"
    
    async def _evaluate_results_with_llm(self, task: str, results: Any, success: bool) -> Dict[str, Any]:
        """Use LLM to evaluate the results and provide feedback"""
        evaluation_prompt = f"""
        Evaluate the results of this task execution:
        
        ORIGINAL TASK: {task}
        EXECUTION SUCCESS: {success}
        RESULTS: {str(results)[:1000]}...
        
        Provide a JSON response with:
        {{
            "task_completed": true/false,
            "quality_score": 0.8,
            "lessons_learned": ["insight1", "insight2"],
            "improvements": ["suggestion1", "suggestion2"],
            "success_factors": ["factor1", "factor2"]
        }}
        
        Return ONLY valid JSON.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "task_completed": success, 
                "quality_score": 0.5, 
                "lessons_learned": [], 
                "improvements": [], 
                "success_factors": []
            }
    
    async def execute_task(self, task: str, use_openai_tool_calling: bool = True) -> Dict[str, Any]:
        """Main method to execute a task autonomously"""
        logger.info(f"Starting task execution: {task}")
        
        try:
            # Ensure servers are initialized
            await self.initialize_servers()
            
            if use_openai_tool_calling:
                # Use OpenAI's built-in tool calling
                results, execution_success, tools_used = await self._execute_with_openai_tool_calling(task)
            else:
                # Use custom planning approach
                plan = await self._plan_task_execution(task)
                logger.info(f"Task planning completed with confidence: {plan.get('confidence', 0)}")
                results, execution_success = await self._execute_plan(plan)
                tools_used = [step.get('tool_name', '') for step in plan.get('plan', []) if step.get('tool_name')]
            
            # Evaluate results with LLM
            evaluation = await self._evaluate_results_with_llm(task, results, execution_success)
            
            # Update memory
            final_success = evaluation.get('task_completed', False)
            self.memory.add_task(task, results, tools_used, final_success)
            
            # Learn from the experience
            if evaluation.get('lessons_learned'):
                for lesson in evaluation['lessons_learned']:
                    self.memory.learned_patterns[task[:50]] = lesson
            
            return {
                'task': task,
                'success': final_success,
                'results': results,
                'evaluation': evaluation,
                'tools_used': tools_used
            }
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            self.memory.add_task(task, str(e), [], False)
            return {
                'task': task,
                'success': False,
                'results': str(e),
                'error': True
            }
        
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# Global agent instance
agent = None

def format_output(result: Dict[str, Any]) -> str:
    """Format the output for better display in Gradio"""
    output = f"## Task: {result['task']}\n\n"
    output += f"**Success:** {'✅ Yes' if result['success'] else '❌ No'}\n\n"
    
    if result.get('tools_used'):
        output += f"**Tools Used:** {', '.join(result['tools_used'])}\n\n"
    
    output += f"### Results:\n"
    if isinstance(result['results'], list):
        for i, res in enumerate(result['results'], 1):
            output += f"{i}. {str(res)}\n"
    else:
        output += f"{str(result['results'])}\n"
    
    if 'evaluation' in result and result['evaluation']:
        eval_data = result['evaluation']
        output += f"\n### Evaluation:\n"
        output += f"- **Quality Score:** {eval_data.get('quality_score', 'N/A')}\n"
        output += f"- **Task Completed:** {'✅ Yes' if eval_data.get('task_completed', False) else '❌ No'}\n"
        
        if eval_data.get('lessons_learned'):
            output += f"- **Lessons Learned:** {', '.join(eval_data['lessons_learned'])}\n"
        
        if eval_data.get('success_factors'):
            output += f"- **Success Factors:** {', '.join(eval_data['success_factors'])}\n"
    
    if result.get('error'):
        output += f"\n### Error:\n{result['results']}"
    
    return output

def execute_task_sync(task: str, api_key: str, use_openai_calling: bool = True) -> str:
    """Synchronous wrapper for the async execute_task method"""
    global agent
    
    if not task.strip():
        return "❌ Please enter a task to execute."
    
    if not api_key.strip():
        return "❌ Please enter your OpenAI API key."
    
    try:
        # Initialize agent if not already done or if API key changed
        if agent is None or agent.openai_client.api_key != api_key:
            server_configs = [
                {
                    'name': 'duckdb',
                    'command': 'python',
                    'args': ["C:\\Anuj\\AI\\MCP\\mcp-servers\\duckdb\\server.py"]
                },
                {
                    'name': 'financial',
                    'command': 'python',
                    'args': ["C:\\Anuj\\AI\\MCP\\mcp-servers\\financial_datasets\\server.py"]
                }
            ]
            agent = MCPAgent(openai_api_key=api_key, server_configs=server_configs)
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(agent.execute_task(task, use_openai_calling))
            return format_output(result)
        finally:
            loop.close()
            
    except Exception as e:
        return f"❌ Error executing task: {str(e)}"

def get_agent_status() -> str:
    """Get the current status of the agent"""
    global agent
    
    if agent is None:
        return "🔴 Agent not initialized"
    
    if not agent.initialized:
        return "🟡 Agent initializing..."
    
    status = "🟢 Agent ready\n\n"
    status += f"**Connected Servers:** {len(agent.sessions)}\n"
    status += f"**Available Tools:** {len(agent.available_tools_list)}\n"
    status += f"**Tasks Completed:** {len(agent.memory.task_history)}\n"
    
    if agent.available_tools:
        status += "\n**Available Tools by Server:**\n"
        for server, tools in agent.available_tools.items():
            status += f"- **{server}:** {', '.join(tools.keys())}\n"
    
    return status

def get_task_history() -> str:
    """Get the task history from agent memory"""
    global agent
    
    if agent is None or not agent.memory.task_history:
        return "No task history available."
    
    history = "## Task History\n\n"
    for i, task in enumerate(reversed(agent.memory.task_history[-10:]), 1):
        status = "✅" if task['success'] else "❌"
        history += f"{i}. {status} **{task['task'][:50]}{'...' if len(task['task']) > 50 else ''}**\n"
        history += f"   - *{task['timestamp']}*\n"
        history += f"   - Tools: {', '.join(task['tools_used']) if task['tools_used'] else 'None'}\n\n"
    
    return history

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(
        title="MCP Agent Interface",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .task-output {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 MCP Agent Interface
        
        This interface allows you to interact with the MCP (Model Context Protocol) Agent.
        The agent can execute tasks using various tools and servers.
        """)
        
        with gr.Tab("Execute Task"):
            with gr.Row():
                with gr.Column(scale=2):
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="Enter your OpenAI API key",
                        type="password",
                        value=os.getenv('OPENAI_API_KEY', '')
                    )
                    
                    task_input = gr.Textbox(
                        label="Task",
                        placeholder="Enter the task you want the agent to execute...",
                        lines=3
                    )
                    
                    with gr.Row():
                        use_openai_checkbox = gr.Checkbox(
                            label="Use OpenAI Tool Calling",
                            value=True,
                            info="Use OpenAI's built-in tool calling (recommended)"
                        )
                        
                        execute_btn = gr.Button("🚀 Execute Task", variant="primary")
                
                with gr.Column(scale=1):
                    status_output = gr.Markdown(
                        label="Agent Status",
                        value="🔴 Agent not initialized"
                    )
                    
                    refresh_status_btn = gr.Button("🔄 Refresh Status")
            
            task_output = gr.Markdown(
                label="Task Results",
                elem_classes=["task-output"]
            )
        
        with gr.Tab("Task History"):
            history_output = gr.Markdown(
                label="Recent Tasks",
                value="No task history available."
            )
            refresh_history_btn = gr.Button("🔄 Refresh History")
        
        with gr.Tab("Examples"):
            gr.Markdown("""
            ## Example Tasks
            
            Here are some example tasks you can try:
            
            ### Database Operations
            - "Create a table called 'employees' with columns: id, name, department, salary"
            - "Insert some sample employee data into the employees table"
            - "Show me all employees in the engineering department"
            - "Calculate the average salary by department"
            
            ### Weather Queries (if weather server is configured)
            - "What's the weather forecast for New York?"
            - "Are there any weather alerts for California?"
            
            ### Financial Data (if financial server is configured)
            - "Get the current stock price of Apple (AAPL)"
            - "Show me the income statement for Microsoft"
            
            ### General Queries
            - "List all available tools"
            - "What databases are currently available?"
            - "Help me understand what this agent can do"
            """)
        
        # Event handlers
        execute_btn.click(
            fn=execute_task_sync,
            inputs=[task_input, api_key_input, use_openai_checkbox],
            outputs=[task_output]
        )
        
        refresh_status_btn.click(
            fn=get_agent_status,
            outputs=[status_output]
        )
        
        refresh_history_btn.click(
            fn=get_task_history,
            outputs=[history_output]
        )
        
        # Auto-refresh status on page load
        demo.load(
            fn=get_agent_status,
            outputs=[status_output]
        )
    
    return demo

# Main function to run Gradio
def main():
    """Main function to run the Gradio interface"""
    print("Starting MCP Agent Gradio Interface...")
    
    # Set environment variable if not set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("You'll need to enter your API key in the interface.")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with share=True to get a public link
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want a public link
        debug=True
    )

if __name__ == "__main__":
    main()